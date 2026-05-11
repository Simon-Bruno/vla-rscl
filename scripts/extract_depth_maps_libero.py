# extract per-frame depth maps from libero episode videos (both cameras)
# saves grayscale pngs: {suite}/depth_maps/{camera}/episode_XXXXXX_FFFFFF.png
#
# usage:
#   python scripts/extract_depth_maps_libero.py \
#       --data_root data/libero \
#       --batch_size 64

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


CAMERAS = ["observation.images.image", "observation.images.wrist_image"]


def frames_from_video(path: Path) -> list[np.ndarray]:
    # use torchvision to handle AV1-encoded videos (opencv can't decode them)
    try:
        import torchvision
        reader = torchvision.io.VideoReader(str(path), "video")
        frames = []
        for frame in reader:
            img = frame["data"].permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
            frames.append(img)
        return frames
    except Exception:
        # fallback to opencv
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames


def depth_to_uint8(depth: torch.Tensor) -> np.ndarray:
    d = depth.cpu().float()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return (d * 255).numpy().astype(np.uint8)


def process_video(video_path: Path, out_dir: Path, model, processor, device, batch_size: int):
    frames = frames_from_video(video_path)
    if not frames:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem  # e.g. episode_000000

    frame_idx = 0
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        inputs = processor(
            images=[Image.fromarray(f) for f in batch],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            depths = model(**inputs).predicted_depth  # (B, H, W)

        for depth in depths:
            png_path = out_dir / f"{stem}_{frame_idx:06d}.png"
            gray = depth_to_uint8(depth)
            cv2.imwrite(str(png_path), gray)
            frame_idx += 1

    return frame_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/libero")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_id", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  model: {args.model_id}")

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device).eval()
    print("model ready\n")

    data_root = Path(args.data_root)
    suite_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    print(f"{len(suite_dirs)} suite folders found")

    total_frames = 0
    for suite_dir in suite_dirs:
        for camera in CAMERAS:
            # find all video chunks
            video_base = suite_dir / "videos"
            if not video_base.exists():
                continue

            # iterate over chunks (chunk-000, chunk-001, ...)
            chunks = sorted(d for d in video_base.iterdir() if d.is_dir())
            for chunk_dir in chunks:
                video_dir = chunk_dir / camera
                if not video_dir.exists():
                    continue

                cam_short = camera.split(".")[-1]  # "image" or "wrist_image"
                out_dir = suite_dir / "depth_maps" / cam_short

                videos = sorted(video_dir.glob("episode_*.mp4"))
                print(f"\n{suite_dir.name} / {cam_short} / {chunk_dir.name}  ({len(videos)} episodes)")

                for video_path in videos:
                    n = process_video(video_path, out_dir, model, processor, device, args.batch_size)
                    total_frames += n
                    print(f"  {video_path.name}  ->  {n} frames")

    print(f"\nall done — {total_frames:,} depth maps saved across both cameras")


if __name__ == "__main__":
    main()
