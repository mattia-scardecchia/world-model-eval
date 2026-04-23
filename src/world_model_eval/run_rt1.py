"""Runs inference with a RT-1 model."""

import copy
from collections import deque

import mediapy as media
import numpy as np
from PIL import Image
from absl import app
from absl import flags
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_hub as hub
import torch
from tqdm import tqdm
from pathlib import Path

from .rt1 import rt1
from .world_model import WorldModel
from .utils import (
    rescale_bridge_action,
    discover_trials,
    predict,
    aggregate_model_results,
    print_results_table,
)

CHECKPOINTS_TO_KWARGS = {
    "bridge_v2_ckpt.pt": {
        "use_pixel_rope": True,
    },
    "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt": {
        "use_pixel_rope": False,
        "default_cfg": 3.0,
    },
}

_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Path to RT-1 JAX checkpoint directory.",
)
_WORLD_MODEL_CKPT = flags.DEFINE_string(
    "world_model_checkpoint",
    None,
    "Path to world-model checkpoint (.pt).",
)
_ROOT_DIR = flags.DEFINE_string(
    "root_dir",
    None,
    "Directory containing evaluation trials.",
)
_ROLLOUT_LENGTH = flags.DEFINE_integer(
    "rollout_length",
    40,
    "Number of simulation steps per rollout.",
)
_RETRIES = flags.DEFINE_integer(
    "retries",
    1,
    "Number of retries per trial.",
)
_SAVE_VIDEO = flags.DEFINE_bool(
    "save_video",
    False,
    "Whether to save rollout videos.",
)
_VIDEO_OUT_DIR = flags.DEFINE_string(
    "video_out_dir",
    None,
    "Directory to store rollout videos when --save_video is set.",
)

flags.mark_flag_as_required("checkpoint_path")
flags.mark_flag_as_required("world_model_checkpoint")
flags.mark_flag_as_required("root_dir")


def _configure_tensorflow():
  """Force TensorFlow ops to run on CPU to avoid CUDA handle issues."""
  try:
    tf.config.set_visible_devices([], "GPU")
  except (RuntimeError, ValueError):
    # Happens if GPUs are already initialized or unavailable; safe to ignore.
    pass

# Optional - can remove
_configure_tensorflow()

# Load the sentence encoder once
_USE = None
def get_use():
    global _USE
    if _USE is None:
        _USE = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
    return _USE


class RT1Policy:
  """Runs inference with a RT-1 policy."""

  def __init__(
      self,
      checkpoint_path=None,
      model=rt1.RT1(),
      variables=None,
      seqlen=15,
      rng=None,
  ):
    if not variables and not checkpoint_path:
      raise ValueError('At least one of `variables` or `checkpoint_path` must be defined.')
    self.model = model
    self._checkpoint_path = checkpoint_path
    self.seqlen = seqlen

    self._run_action_inference_jit = jax.jit(self._run_action_inference)

    self.rng = jax.random.PRNGKey(0) if rng is None else rng

    if variables:
      self.variables = variables
    else:
      state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
      variables = {
          'params': state_dict['params'],
          'batch_stats': state_dict['batch_stats'],
      }
      self.variables = variables

  def _run_action_inference(self, observation, rng):
    """A jittable function for running inference."""

    act_tokens = jnp.zeros((1, 6, 11))


    batch_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

    _, random_rng = jax.random.split(rng)

    output_logits = self.model.apply(
        self.variables,
        batch_obs,
        act=None,
        act_tokens=act_tokens,
        train=False,
        rngs={'random': random_rng},
    )

    time_step_tokens = self.model.num_image_tokens + self.model.num_action_tokens
    output_logits = jnp.reshape(output_logits, (1, self.seqlen, time_step_tokens, -1))
    action_logits = output_logits[:, -1, ...]
    action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

    action_logp = jax.nn.softmax(action_logits)
    action_token = jnp.argmax(action_logp, axis=-1)

    detokenized = rt1.detokenize_action(
        action_token, self.model.vocab_size, self.model.world_vector_range
    )

    detokenized = jax.tree_util.tree_map(lambda x: x[0], detokenized)
    return detokenized

  def action(self, observation):
    """Outputs the action given observation from the env."""

    observation = copy.deepcopy(observation)

    observation.pop('natural_language_instruction', None)

    image = observation['image']  # [T, H, W, 3], uint8 or float
    # Resize with TF to match training; scale to [0,1].
    image = tf.image.resize(image, (300, 300)).numpy()
    image = image / 255.0 if image.max() > 1.0 else image
    observation['image'] = image.astype(np.float32)

    self.rng, rng = jax.random.split(self.rng)
    action = self._run_action_inference_jit(observation, rng)
    action = jax.device_get(action)

    # Ensure a valid terminate token
    if np.sum(action.get('terminate_episode', np.array([0]))) == 0:
      te = np.zeros_like(action['terminate_episode'])
      te[-1] = 1
      action['terminate_episode'] = te
    return action


def _build_rt1_observation(frame_hist, instr_embed):
  """
  frame_hist: deque/list of length T with uint8 HxWx3 images
  instr_embed: np.ndarray (512,) USE embedding for instruction
  Returns observation dict with:
    - image: jnp array [T, 300, 300, 3] (float32 in [0,1] will be ensured in policy)
    - natural_language_embedding: jnp array [T, 512]
  """
  T = len(frame_hist)
  imgs = np.stack(frame_hist, axis=0)  # [T, H, W, 3], uint8
  nle = np.tile(instr_embed[None, :], (T, 1))  # [T, 512]
  return {
      'image': jnp.array(imgs),
      'natural_language_embedding': jnp.array(nle, dtype=jnp.float32),
  }


def evaluate_rt1(wm: WorldModel, policy: RT1Policy, trials, rollout_length=40, retries=1,
                 history_len=15, save_video=False, video_out_dir=None, root_dir=None,
                 scorer_n=5, return_raw=False):
  """
  Evaluate RT-1 on discovered trials using the world model. Returns a list of per-trial dicts with 'score'.
  When return_raw=True, also returns a flat scorer_log list with per-scorer-call rows.
  """
  results = []
  scorer_log = []
  if save_video and video_out_dir:
    Path(video_out_dir).mkdir(parents=True, exist_ok=True)

  use = get_use()

  with torch.no_grad():
    for trial_idx, trial in enumerate(tqdm(trials, desc="RT-1 trials")):
      start_frame = np.array(Image.open(trial["trial_png"]).resize((256, 256)))

      emb = use([trial['instruction']]).numpy()[0].astype(np.float32)

      for r in range(retries):
        wm.reset(torch.from_numpy(start_frame).cuda().float() / 255.0)
        frames = [start_frame]

        hist = deque([start_frame] * history_len, maxlen=history_len)

        for step in range(rollout_length):

          obs = _build_rt1_observation(hist, emb)
          detok = policy.action(obs)

          wv   = np.asarray(detok.get('world_vector', None), dtype=np.float32)
          rot  = np.asarray(detok.get('rotation_delta', None), dtype=np.float32)
          grip = np.asarray(detok.get('gripper_closedness_action', None), dtype=np.float32)
          if wv is None or rot is None or grip is None:
            raise RuntimeError("RT-1 detokenized action missing required keys.")
          if wv.shape[0] < 3 or rot.shape[0] < 3:
            raise RuntimeError(f"Unexpected RT-1 action shapes: world_vector {wv.shape}, rotation_delta {rot.shape}")

          # Normalize: world_vector in [-2,2] -> divide by 2; rotation in [-pi/2,pi/2] -> divide by (pi/2)
          x, y, z = (wv[:3] / 2.0).tolist()
          # x, y, z = (wv[:3]).tolist()
          roll, pitch, yaw = (rot[:3] / (np.pi / 2)).tolist()
          # roll, pitch, yaw = (rot[:3] / np.pi).tolist()
          gripper = float(grip.reshape(-1)[0])  # already in [-1,1]
          a7 = torch.tensor([x, y, z, roll, pitch, yaw, gripper], device="cuda", dtype=torch.float32)

          a10 = torch.cat([a7, a7.new_zeros(3)], dim=-1)
          a10 = rescale_bridge_action(a10, wv_lo=-1, wv_hi=1, rd_lo=-1, rd_hi=1)

          for _, x in wm.generate_chunk(a10):
            new_frame = x[0, 0].cpu().numpy()
            new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
            frames.append(new_frame)
            hist.append(new_frame)

        rollout_video = np.stack(frames)
        video_out_path = None
        if save_video and video_out_dir:
          trial_png = Path(trial["trial_png"])
          rel_parent = (
              trial_png.parent
              if root_dir is None
              else trial_png.parent.relative_to(Path(root_dir))
          )
          target_dir = Path(video_out_dir) / rel_parent
          target_dir.mkdir(parents=True, exist_ok=True)
          stem = trial_png.stem
          out_name = f"{stem}.mp4"
          video_out_path = str(target_dir / out_name)
          media.write_video(video_out_path, rollout_video, fps=20)

        if return_raw:
          score, per_call = predict(rollout_video, trial, n=scorer_n, return_raw=True)
          for call_idx, call in enumerate(per_call):
            scorer_log.append({
                "task_key": trial["task_key"],
                "task_display": trial["task_display"],
                "trial_idx": trial_idx,
                "trial_png": str(trial["trial_png"]),
                "retry_idx": r,
                "scorer_call_idx": call_idx,
                "video_path": video_out_path,
                "score": call["parsed_score"],
                "rationale": call["rationale"],
                "openai_meta": {
                    "model": call["model"],
                    "finish_reason": call["finish_reason"],
                    "prompt_tokens": call["prompt_tokens"],
                    "completion_tokens": call["completion_tokens"],
                    "response_id": call["response_id"],
                },
            })
        else:
          score = predict(rollout_video, trial, n=scorer_n)
        results.append({
            "task_key": trial["task_key"],
            "task_display": trial["task_display"],
            "score": float(score),
        })
  if return_raw:
    return results, scorer_log
  return results


def _absl_main(argv):
  del argv
  # RT-1-X config
  sequence_length = 15
  num_action_tokens = 11
  layer_size = 256
  vocab_size = 512
  num_image_tokens = 81
  rt1x_model = rt1.RT1(
      num_image_tokens=num_image_tokens,
      num_action_tokens=num_action_tokens,
      layer_size=layer_size,
      vocab_size=vocab_size,
      use_token_learner=True,
      world_vector_range=(-2.0, 2.0),
  )
  policy = RT1Policy(
      checkpoint_path=_CHECKPOINT_PATH.value,
      model=rt1x_model,
      seqlen=sequence_length,
  )

  checkpoint_path = Path(_CHECKPOINT_PATH.value)
  if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"RT-1 checkpoint not found at {_CHECKPOINT_PATH.value}. Provide a valid path."
    )

  wm_path = Path(_WORLD_MODEL_CKPT.value)
  if not wm_path.exists():
    raise FileNotFoundError(
        f"World model checkpoint not found at {_WORLD_MODEL_CKPT.value}. "
        "Pass --world_model_checkpoint with a valid file."
    )

  root_dir = _ROOT_DIR.value
  if root_dir is None:
    raise ValueError("root_dir must be provided via --root_dir to locate evaluation trials.")

  wm = WorldModel(wm_path, **CHECKPOINTS_TO_KWARGS.get(wm_path.name, {}))

  trials = discover_trials(root_dir)
  print(f"Discovered {len(trials)} trials.")

  results = evaluate_rt1(
      wm,
      policy,
      trials,
      rollout_length=_ROLLOUT_LENGTH.value,
      retries=_RETRIES.value,
      history_len=sequence_length,
      save_video=_SAVE_VIDEO.value,
      video_out_dir=_VIDEO_OUT_DIR.value,
      root_dir=root_dir,
  )

  agg = aggregate_model_results(results)
  print_results_table(agg)


def main():
  app.run(_absl_main)


if __name__ == '__main__':
  main()
