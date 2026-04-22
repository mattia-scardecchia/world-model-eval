import argparse
from typing import Sequence

from .utils import (
    rescale_bridge_action,
    discover_trials,
    predict,
    aggregate_model_results,
    print_results_table,
)
from transformers import AutoProcessor, AutoModel
from .world_model import WorldModel
import numpy as np
from PIL import Image
import mediapy as media
from tqdm import tqdm
import torch
from pathlib import Path

def normalize_actions(unnorm_actions, statistics, key="bridge_orig/1.0.0"):
    stats = statistics[key]["action"]
    action_low = np.array(stats["q01"])
    action_high = np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(action_low)), dtype=bool)
    norm_actions = np.where(
        mask,
        2 * (unnorm_actions - action_low) / (action_high - action_low) - 1,
        unnorm_actions,
    )
    return norm_actions

def evaluate_spatialvla(wm, vla, processor, trials, retries=1, rollout_length=40,
                        save_video=False, video_out_dir=None, root_dir=None):
    """
    Roll out SpatialVLA on a list of trials discovered from ROOT_DIR and return per-trial scores.
    """
    results = []
    if save_video and video_out_dir:
        Path(video_out_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for trial in tqdm(trials, desc="SpatialVLA trials"):
            start_frame = np.array(Image.open(trial["trial_png"]).resize((256, 256)))
            for r in range(retries):
                wm.reset(torch.from_numpy(start_frame).cuda().float() / 255.0)
                frames = [start_frame]
                for step in range(rollout_length):
                    curr_frame = Image.fromarray(frames[-1])
                    prompt = f"In: What action should the robot take to {trial['instruction']}?\nOut:"
                    inputs = processor(images=[curr_frame], text=prompt, return_tensors="pt").to(
                        device="cuda", dtype=torch.bfloat16
                    )
                    generation_outputs = vla.predict_action(inputs)
                    actions = processor.decode_actions(
                        generation_outputs, unnorm_key="bridge_orig/1.0.0"
                    )["actions"]
                    seq_len = actions.shape[0]
                    action_chunk = torch.zeros((seq_len, 10), device="cuda", dtype=torch.float32)
                    for ai in range(seq_len):
                        raw = actions[ai]
                        raw = normalize_actions(raw, processor.statistics)
                        a = torch.tensor(raw, device="cuda", dtype=torch.float32)
                        a = torch.cat([a, a.new_zeros(3)], dim=-1)
                        a = rescale_bridge_action(a, wv_lo=-1, wv_hi=1, rd_lo=-1, rd_hi=1)
                        action_chunk[ai] = a
                    if getattr(wm, "chunk_size", None) != seq_len:
                        print(f"Setting wm.chunk_size = {seq_len}")
                        wm.chunk_size = seq_len
                    for _, x in wm.generate_chunk(action_chunk):
                        new_frame = x[0, 0].cpu().numpy()
                        new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
                        frames.append(new_frame)
                rollout_video = np.stack(frames)
                if save_video and video_out_dir:
                    trial_png = Path(trial["trial_png"])
                    target_dir = Path(video_out_dir)
                    if root_dir is not None:
                        try:
                            rel_parent = trial_png.parent.relative_to(Path(root_dir))
                            target_dir = target_dir / rel_parent
                        except ValueError:
                            target_dir = target_dir / trial_png.parent.name
                    else:
                        target_dir = target_dir / trial_png.parent.name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    vid_name = trial_png.stem
                    out_name = f"{vid_name}.mp4"
                    media.write_video(str(target_dir / out_name), rollout_video, fps=20)

                score = predict(rollout_video, trial)
                results.append({
                    "task_key": trial["task_key"],
                    "task_display": trial["task_display"],
                    "score": float(score),
                })

    return results

CHECKPOINTS_TO_KWARGS = {
    "bridge_v2_ckpt.pt": {
        "use_pixel_rope": True,
    },
    "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt": {
        "use_pixel_rope": False,
        "default_cfg": 3.0,
    },
}


def run(
    checkpoint_path: str = "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt",
    model_name: str = "spatialvla-4b-224-pt",
    root_dir: str | None = None,
    *,
    rollout_length: int = 15,
    retries: int = 1,
    save_video: bool = False,
    video_out_dir: str | None = None,
) -> dict[str, dict[str, float]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}; download it manually and retry.")
    ckpt_key = ckpt_path.name
    ckpt_kwargs = CHECKPOINTS_TO_KWARGS.get(ckpt_key, {})
    wm = WorldModel(ckpt_path, **ckpt_kwargs)

    model_name_or_path = f"IPEC-COMMUNITY/{model_name}"
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    if root_dir is None:
        raise ValueError("root_dir must be provided; pass --root-dir to point at the evaluation dataset.")
    trials = discover_trials(root_dir)
    print(f"Discovered {len(trials)} trials.")

    results = evaluate_spatialvla(
        wm,
        model,
        processor,
        trials,
        rollout_length=rollout_length,
        retries=retries,
        save_video=save_video,
        video_out_dir=video_out_dir,
        root_dir=root_dir,
    )

    agg = aggregate_model_results(results)
    print_results_table(agg)
    return agg


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a SpatialVLA policy in the Bridge world model")
    parser.add_argument("--checkpoint-path", default="mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt")
    parser.add_argument("--model-name", default="spatialvla-4b-224-pt")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--rollout-length", type=int, default=15)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-out-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, dict[str, float]]:  # pragma: no cover - CLI entry point
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return run(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        root_dir=args.root_dir,
        rollout_length=args.rollout_length,
        retries=args.retries,
        save_video=args.save_video,
        video_out_dir=args.video_out_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
