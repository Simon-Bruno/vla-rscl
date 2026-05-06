# extract per-frame depth feature vectors from robocasa episode videos
# uses depth anything v2 small (huggingface) as a frozen feature extractor
# output: data/robocasa/{task}/depth_features/episode_XXXXXX.npy
#         shape: (n_frames, 64) — 8x8 spatially-pooled normalised depth map
#
# usage:
#   python scripts/extract_depth_features.py --data_root data/robocasa --n_episodes 12

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def load_video_frames(video_path: Path) -> list:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_features(frames: list, model, processor, device, batch_size: int = 64) -> np.ndarray:
    all_features = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        pil_images = [Image.fromarray(f) for f in batch]
        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        with torch.no_grad():
            depth = model(**inputs).predicted_depth  # (B, H, W)

        # normalize each depth map to [0, 1]
        b = depth.shape[0]
        d_flat = depth.view(b, -1)
        d_min = d_flat.min(dim=1).values[:, None, None]
        d_max = d_flat.max(dim=1).values[:, None, None]
        depth = (depth - d_min) / (d_max - d_min + 1e-6)

        # spatial pool to 8x8 grid -> 64-d vector per frame
        feat = F.adaptive_avg_pool2d(depth.unsqueeze(1), (8, 8))  # (B, 1, 8, 8)
        feat = feat.squeeze(1).view(b, -1).cpu().numpy()          # (B, 64)
        all_features.append(feat)

    return np.concatenate(all_features, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/robocasa")
    parser.add_argument("--n_episodes", type=int, default=12,
                        help="episodes to process per task (match max_demos_total // n_tasks)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_id", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading {args.model_id} ...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()
    print("model loaded")

    data_root = Path(args.data_root)
    task_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    print(f"found {len(task_dirs)} tasks")

    total_frames = 0
    for task_dir in task_dirs:
        video_dir = task_dir / "videos" / "chunk-000" / "observation.images.ego_view"
        if not video_dir.exists():
            continue

        feat_dir = task_dir / "depth_features"
        feat_dir.mkdir(exist_ok=True)

        episodes = sorted(video_dir.glob("episode_*.mp4"))[: args.n_episodes]
        print(f"\n{task_dir.name}: {len(episodes)} episodes")

        for ep_path in episodes:
            ep_id = ep_path.stem  # episode_000000
            out_path = feat_dir / f"{ep_id}.npy"

            if out_path.exists():
                print(f"  {ep_id} already done, skipping")
                continue

            frames = load_video_frames(ep_path)
            if not frames:
                print(f"  {ep_id} no frames, skipping")
                continue

            features = extract_features(frames, model, processor, device, args.batch_size)
            np.save(out_path, features)
            total_frames += len(frames)
            print(f"  {ep_id}: {len(frames)} frames -> {features.shape}  saved to {out_path}")

    print(f"\ndone. total frames processed: {total_frames}")
    print(f"features saved as float32 npy, shape (n_frames, 64) per episode")


if __name__ == "__main__":
    main()
