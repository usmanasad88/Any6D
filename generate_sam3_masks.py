"""
Generate object masks for all video frames using SAM3 text-prompted segmentation.

Handles multiple instances of the same object class (e.g. two bottles) by
tracking detections across frames via bbox-center proximity.

Usage:
    conda activate sam3
    python generate_sam3_masks.py \
        --video demo_data/bottle/bottlevid.mp4 \
        --prompts bottle scale \
        --instances bottle:2 \
        --skip_frames 5 \
        --output_dir results/sam3_masks \
        --confidence 0.3
"""
from __future__ import annotations

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

# SAM3 imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def extract_frames(video_path: str, skip: int = 1):
    """Yield (frame_index, bgr_frame) from a video file."""
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            yield idx, frame
        idx += 1
    cap.release()


def bbox_center(box):
    """Return (cx, cy) from [x0, y0, x1, y1]."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def assign_detections_to_tracks(det_boxes, prev_centers, max_dist=400):
    """Hungarian matching of detections to previous track centers.

    Args:
        det_boxes: (N, 4) array of [x0,y0,x1,y1]
        prev_centers: dict mapping track_id → (cx, cy) from previous frame
        max_dist: maximum distance for a valid assignment

    Returns:
        assignments: dict mapping track_id → detection_index (or None)
    """
    track_ids = list(prev_centers.keys())
    if len(track_ids) == 0 or len(det_boxes) == 0:
        return {}

    n_tracks = len(track_ids)
    n_dets = len(det_boxes)

    # Build cost matrix (track × detection) using Euclidean distance
    cost = np.full((n_tracks, n_dets), 1e6)
    for i, tid in enumerate(track_ids):
        px, py = prev_centers[tid]
        for j in range(n_dets):
            cx, cy = bbox_center(det_boxes[j])
            cost[i, j] = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            assignments[track_ids[r]] = c

    return assignments


def save_mask_and_meta(output_dir, track_name, frame_count, mask, score, box,
                       n_detections, frame_idx):
    """Save a binary mask and metadata for one track."""
    obj_dir = os.path.join(output_dir, track_name)
    os.makedirs(obj_dir, exist_ok=True)

    mask_uint8 = mask.astype(np.uint8) * 255
    mask_path = os.path.join(obj_dir, f"mask_{frame_count:05d}.png")
    cv2.imwrite(mask_path, mask_uint8)

    meta_path = os.path.join(obj_dir, f"meta_{frame_count:05d}.txt")
    with open(meta_path, "w") as f:
        f.write(f"score={score:.4f}\n")
        f.write(f"bbox={box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}\n")
        f.write(f"n_detections={n_detections}\n")
        f.write(f"frame_idx={frame_idx}\n")

    return mask_uint8.sum() // 255


def save_empty_mask(output_dir, track_name, frame_count, H, W):
    """Save an empty (all-zero) mask."""
    obj_dir = os.path.join(output_dir, track_name)
    os.makedirs(obj_dir, exist_ok=True)
    mask_path = os.path.join(obj_dir, f"mask_{frame_count:05d}.png")
    cv2.imwrite(mask_path, np.zeros((H, W), dtype=np.uint8))
    meta_path = os.path.join(obj_dir, f"meta_{frame_count:05d}.txt")
    with open(meta_path, "w") as f:
        f.write("score=0.0\nbbox=0,0,0,0\n")


def main():
    parser = argparse.ArgumentParser(description="Generate SAM3 masks for video frames")
    parser.add_argument("--video", type=str, default="demo_data/bottle/bottlevid.mp4",
                        help="Path to input video")
    parser.add_argument("--prompts", type=str, nargs="+", default=["bottle", "scale"],
                        help="Text prompts for objects to segment")
    parser.add_argument("--instances", type=str, nargs="*", default=[],
                        help="Number of instances per prompt, e.g. bottle:2 scale:1. "
                             "Prompts not listed default to 1 instance.")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Process every N-th frame (must match pipeline skip_frames)")
    parser.add_argument("--output_dir", type=str, default="results/sam3_masks",
                        help="Directory to save masks")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold for SAM3 detections")
    parser.add_argument("--first_frame_only", action="store_true",
                        help="Only generate masks for the first frame")
    args = parser.parse_args()

    # Parse instance counts
    instance_counts = {}
    for spec in args.instances:
        prompt, count = spec.split(":")
        instance_counts[prompt] = int(count)
    for prompt in args.prompts:
        instance_counts.setdefault(prompt, 1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build track names: for prompts with >1 instance, name them prompt_0, prompt_1, …
    # For single-instance prompts, just use the prompt name
    track_info = {}   # track_name → prompt
    for prompt in args.prompts:
        n = instance_counts[prompt]
        if n == 1:
            track_info[prompt] = prompt
        else:
            for i in range(n):
                track_info[f"{prompt}_{i}"] = prompt

    for track_name in track_info:
        os.makedirs(os.path.join(args.output_dir, track_name), exist_ok=True)

    print(f"[INFO] Tracks to generate: {list(track_info.keys())}")

    # ── Build SAM3 model ────────────────────────────────────────────────
    print("[INFO] Building SAM3 model …")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    print("[INFO] SAM3 model ready.")

    # ── Process video frames ────────────────────────────────────────────
    print(f"[INFO] Processing video: {args.video} (skip={args.skip_frames})")

    # Per-prompt tracking state: prev_centers[prompt] = {track_name: (cx, cy)}
    prev_centers = {}

    frame_count = 0
    for idx, bgr in extract_frames(args.video, skip=args.skip_frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        pil_image = Image.fromarray(rgb)

        state = processor.set_image(pil_image)

        for prompt in args.prompts:
            n_inst = instance_counts[prompt]
            track_names = [t for t, p in track_info.items() if p == prompt]

            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)

            masks = state["masks"]     # (N, 1, H, W)
            scores = state["scores"]   # (N,)
            boxes = state["boxes"]     # (N, 4)

            n_dets = len(scores)
            if n_dets == 0:
                print(f"  Frame {idx}: No detections for '{prompt}'")
                for tname in track_names:
                    save_empty_mask(args.output_dir, tname, frame_count, H, W)
                continue

            # Convert to numpy (cast from bfloat16 → float32 first)
            det_masks = masks[:, 0].cpu().float().numpy()     # (N, H, W)
            det_scores = scores.cpu().float().numpy()          # (N,)
            det_boxes = boxes.cpu().float().numpy()            # (N, 4)

            # Binarize masks (threshold at 0.5)
            det_masks = (det_masks > 0.5).astype(bool)

            if n_inst == 1:
                # Simple case: pick highest-score detection
                best = det_scores.argmax()
                tname = track_names[0]
                npix = save_mask_and_meta(
                    args.output_dir, tname, frame_count,
                    det_masks[best], det_scores[best], det_boxes[best],
                    n_dets, idx,
                )
                cx, cy = bbox_center(det_boxes[best])
                prev_centers.setdefault(prompt, {})[tname] = (cx, cy)
                print(f"  Frame {idx} (#{frame_count}): '{tname}' → "
                      f"score={det_scores[best]:.3f}, "
                      f"center=({cx:.0f},{cy:.0f}), "
                      f"pixels={npix}")
            else:
                # Multi-instance: use Hungarian matching to track identities
                if prompt not in prev_centers or frame_count == 0:
                    # First frame: sort detections by score (descending),
                    # take top-N, assign to tracks by spatial position
                    # Sort by x-coordinate so assignment is deterministic
                    top_n = min(n_inst, n_dets)
                    top_idxs = det_scores.argsort()[::-1][:top_n]
                    # Sort these by x-center so left=0, right=1, etc.
                    centers = [bbox_center(det_boxes[i]) for i in top_idxs]
                    order = np.argsort([c[0] for c in centers])
                    sorted_idxs = top_idxs[order]

                    prev_centers[prompt] = {}
                    for k, tname in enumerate(track_names):
                        if k < len(sorted_idxs):
                            di = sorted_idxs[k]
                            npix = save_mask_and_meta(
                                args.output_dir, tname, frame_count,
                                det_masks[di], det_scores[di], det_boxes[di],
                                n_dets, idx,
                            )
                            cx, cy = bbox_center(det_boxes[di])
                            prev_centers[prompt][tname] = (cx, cy)
                            print(f"  Frame {idx} (#{frame_count}): '{tname}' → "
                                  f"score={det_scores[di]:.3f}, "
                                  f"center=({cx:.0f},{cy:.0f}), "
                                  f"pixels={npix}")
                        else:
                            save_empty_mask(args.output_dir, tname, frame_count, H, W)
                            print(f"  Frame {idx} (#{frame_count}): '{tname}' → NO DETECTION")
                else:
                    # Subsequent frames: match to previous centers
                    assignments = assign_detections_to_tracks(
                        det_boxes, prev_centers[prompt], max_dist=400
                    )
                    for tname in track_names:
                        if tname in assignments:
                            di = assignments[tname]
                            npix = save_mask_and_meta(
                                args.output_dir, tname, frame_count,
                                det_masks[di], det_scores[di], det_boxes[di],
                                n_dets, idx,
                            )
                            cx, cy = bbox_center(det_boxes[di])
                            prev_centers[prompt][tname] = (cx, cy)
                            print(f"  Frame {idx} (#{frame_count}): '{tname}' → "
                                  f"score={det_scores[di]:.3f}, "
                                  f"center=({cx:.0f},{cy:.0f}), "
                                  f"pixels={npix}")
                        else:
                            save_empty_mask(args.output_dir, tname, frame_count, H, W)
                            # Keep previous center for next-frame matching
                            print(f"  Frame {idx} (#{frame_count}): '{tname}' → LOST")

        frame_count += 1
        if args.first_frame_only:
            break

    # Save frame count metadata
    all_tracks = list(track_info.keys())
    with open(os.path.join(args.output_dir, "info.txt"), "w") as f:
        f.write(f"total_frames={frame_count}\n")
        f.write(f"skip_frames={args.skip_frames}\n")
        f.write(f"tracks={','.join(all_tracks)}\n")
        f.write(f"prompts={','.join(args.prompts)}\n")
        f.write(f"video={args.video}\n")
        f.write(f"confidence={args.confidence}\n")
        for spec in args.instances:
            f.write(f"instances_{spec}\n")

    print(f"\n[DONE] Generated masks for {frame_count} frames → {args.output_dir}")
    print(f"  Tracks: {all_tracks}")


if __name__ == "__main__":
    main()
