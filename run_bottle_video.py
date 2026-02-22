"""
6DOF Pose Tracking Pipeline for Known Objects in Monocular Video.

Pipeline overview:
  1. Extract frames from input video (bottlevid.mp4)
  2. Use Depth Anything 3 (DA3) to produce metric depth for each frame
  3. Load pre-generated SAM3 masks (text-prompted) for each object
  4. Load GLB meshes, auto-scale to real-world size using depth + mask OBB
  5. Run Any6D register_any6d on the first frame to initialise 6DOF pose
  6. Run track_one_any6d on subsequent frames
  7. Render textured mesh overlays via nvdiffrast and write output video

Pre-requisite:
    Generate masks first (in sam3 conda env):
        conda activate sam3
        python generate_sam3_masks.py --skip_frames 5 --prompts bottle scale

Usage:
    conda activate Any6D
    python run_bottle_video.py --use_da3_intrinsics --skip_frames 5 \
        --sam3_mask_dir results/sam3_masks
"""
from __future__ import annotations

import os
import copy
import argparse
import yaml
import cv2
import numpy as np
import torch
import trimesh
import imageio
from PIL import Image
from pytorch_lightning import seed_everything

# ── Any6D / FoundationPose imports ──────────────────────────────────────────
import nvdiffrast.torch as dr
from estimater import Any6D
from foundationpose.Utils import (
    nvdiffrast_render,
    make_mesh_tensors,
    draw_xyz_axis,
    draw_posed_3d_box,
)
from foundationpose.learning.training.predict_score import ScorePredictor
from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor

# ── SAM2 imports (kept as fallback) ─────────────────────────────────────
try:
    from sam2.sam2.build_sam import build_sam2
    from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

# ── Depth Anything 3 imports ───────────────────────────────────────────────

from depth_anything_3.api import DepthAnything3


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_glb_mesh(glb_path: str) -> trimesh.Trimesh:
    """Load a GLB mesh and convert PBRMaterial → SimpleMaterial so that
    the FoundationPose renderer can access ``material.image``."""
    mesh = trimesh.load(glb_path, force="mesh")
    if hasattr(mesh.visual, "material"):
        mat = mesh.visual.material
        # PBRMaterial has no .image; convert to SimpleMaterial
        if type(mat).__name__ == "PBRMaterial":
            mesh.visual.material = mat.to_simple()
    return mesh


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


def get_video_info(video_path: str):
    """Return (width, height, fps, total_frames) for a video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    h, w = frame.shape[:2]  # OpenCV frame shape is (H, W, C)
    return w, h, fps, n


def segment_object_sam2(
    rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    sam_ckpt: str = "./sam2/checkpoints/sam2.1_hiera_large.pt",
    sam_cfg: str = "./sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
) -> np.ndarray:
    """Run SAM2 image predictor with a bounding-box prompt.
    Returns a boolean mask (H, W)."""
    if not HAS_SAM2:
        raise RuntimeError("SAM2 not available. Please use --sam3_mask_dir instead.")
    predictor = SAM2ImagePredictor(build_sam2(sam_cfg, sam_ckpt))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox_xyxy[None],  # (1, 4)
            multimask_output=False,
        )
    mask = masks[0].astype(bool)
    del predictor
    torch.cuda.empty_cache()
    return mask


def load_sam3_mask(mask_dir: str, obj_name: str, frame_idx: int) -> np.ndarray | None:
    """Load a pre-generated SAM3 mask from disk.
    Returns a boolean mask (H, W) or None if not found."""
    mask_path = os.path.join(mask_dir, obj_name, f"mask_{frame_idx:05d}.png")
    if not os.path.exists(mask_path):
        return None
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
    return mask_img > 127  # Convert to boolean


def compute_da3_depth(
    da3_model,
    frames_rgb: list,
    intrinsic=None,
):
    """Run DA3 metric depth on a batch of RGB frames.
    Returns depth array (N, H, W) in metres."""
    pil_images = [Image.fromarray(f) for f in frames_rgb]

    # Prepare intrinsics if provided (tile for all frames)
    extrinsics = None
    intrinsics = None
    if intrinsic is not None:
        N = len(pil_images)
        intrinsics = np.tile(intrinsic[None], (N, 1, 1)).astype(np.float32)

    prediction = da3_model.inference(
        image=pil_images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=504,
        process_res_method="upper_bound_resize",
    )
    # prediction.depth is (N, proc_H, proc_W); resize back to original
    depths = prediction.depth  # numpy float32
    return depths, prediction


def resize_depth_to_frame(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear-resize a depth map to (target_h, target_w)."""
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def downscale_for_pose(rgb, depth, mask, K, max_side=480):
    """Downscale inputs for pose estimation to avoid CUDA OOM.
    Returns (rgb_s, depth_s, mask_s, K_s, scale_factor)."""
    H, W = rgb.shape[:2]
    if max(H, W) <= max_side:
        return rgb, depth, mask, K, 1.0
    scale = max_side / max(H, W)
    new_W = int(W * scale)
    new_H = int(H * scale)
    rgb_s = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    depth_s = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask.astype(np.uint8), (new_W, new_H),
                        interpolation=cv2.INTER_NEAREST).astype(bool) if mask is not None else None
    K_s = K.copy()
    K_s[0, :] *= scale
    K_s[1, :] *= scale
    return rgb_s, depth_s, mask_s, K_s, scale


def downscale_rgb_depth(rgb, depth, K, max_side=480):
    """Downscale rgb+depth+K for tracking (no mask needed)."""
    H, W = rgb.shape[:2]
    if max(H, W) <= max_side:
        return rgb, depth, K
    scale = max_side / max(H, W)
    new_W = int(W * scale)
    new_H = int(H * scale)
    rgb_s = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    depth_s = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    K_s = K.copy()
    K_s[0, :] *= scale
    K_s[1, :] *= scale
    return rgb_s, depth_s, K_s


def render_mesh_overlay(
    rgb: np.ndarray,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    K: np.ndarray,
    glctx,
    alpha: float = 0.55,
    mesh_tensors=None,
):
    """Render a textured mesh on top of an RGB image with transparency.
    Returns an RGB uint8 image."""
    H, W = rgb.shape[:2]
    if mesh_tensors is None:
        mesh_tensors = make_mesh_tensors(mesh)
    ob_in_cams = torch.tensor(pose[None], device="cuda", dtype=torch.float32)

    ren_img, ren_depth, _ = nvdiffrast_render(
        K=K, H=H, W=W,
        mesh=mesh,
        ob_in_cams=ob_in_cams,
        context="cuda",
        use_light=True,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        extra={},
    )
    ren_img = (ren_img[0] * 255.0).detach().cpu().numpy().astype(np.uint8)
    ren_depth = ren_depth[0].detach().cpu().numpy()
    ren_mask = ren_depth > 0

    # Blend: overlay where mesh is visible
    out = rgb.copy()
    out[ren_mask] = (
        alpha * ren_img[ren_mask].astype(np.float32)
        + (1 - alpha) * rgb[ren_mask].astype(np.float32)
    ).astype(np.uint8)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    seed_everything(0)

    parser = argparse.ArgumentParser(description="6DOF pose tracking on monocular video")
    parser.add_argument("--video", type=str, default="demo_data/bottle/bottlevid.mp4",
                        help="Path to input video")
    parser.add_argument("--bottle_glb", type=str, default="demo_data/bottle/bottle.glb")
    parser.add_argument("--scale_glb", type=str, default="demo_data/bottle/scale.glb")
    parser.add_argument("--intrinsic_file", type=str,
                        default="demo_data/836212060125_640x480.yml",
                        help="Camera intrinsics YAML (RealSense style)")
    parser.add_argument("--da3_model", type=str, default="da3nested-giant-large",
                        help="DA3 model name for metric depth")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Process every N-th frame")
    parser.add_argument("--est_refine_iter", type=int, default=5,
                        help="Refine iterations for initial registration")
    parser.add_argument("--track_refine_iter", type=int, default=2,
                        help="Refine iterations for frame-to-frame tracking")
    parser.add_argument("--output", type=str, default="results/bottle_tracked.mp4")
    parser.add_argument("--save_dir", type=str, default="results/bottle_tracking")
    parser.add_argument("--da3_batch_size", type=int, default=8,
                        help="Number of frames to process per DA3 batch")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Global depth scale multiplier (tune if needed)")
    parser.add_argument("--use_da3_intrinsics", action="store_true",
                        help="Use DA3's predicted intrinsics instead of calibration file")
    parser.add_argument("--overlay_alpha", type=float, default=0.6,
                        help="Mesh overlay transparency (0 = invisible, 1 = opaque)")
    # SAM3 pre-generated masks directory (preferred)
    parser.add_argument("--sam3_mask_dir", type=str, default=None,
                        help="Directory with pre-generated SAM3 masks (from generate_sam3_masks.py)")
    # Bounding boxes for SAM2 fallback (only needed if --sam3_mask_dir not set)
    parser.add_argument("--bottle_bbox", type=int, nargs=4,
                        default=None,
                        help="Bounding box for the bottle in the first frame (x1 y1 x2 y2)")
    parser.add_argument("--scale_bbox", type=int, nargs=4,
                        default=None,
                        help="Bounding box for the scale in the first frame (x1 y1 x2 y2)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── 1. Read video info ──────────────────────────────────────────────
    W_vid, H_vid, fps, total_frames = get_video_info(args.video)
    print(f"[INFO] Video: {args.video}  {W_vid}×{H_vid} @ {fps:.1f} fps, {total_frames} frames")

    # ── 2. Camera intrinsics ────────────────────────────────────────────
    # Try loading from YAML; if it doesn't match the video resolution,
    # scale the intrinsics or estimate from a reasonable FOV.
    K = None
    if args.intrinsic_file and os.path.exists(args.intrinsic_file):
        with open(args.intrinsic_file, "r") as f:
            cam_data = yaml.load(f, Loader=yaml.FullLoader)
        K_ref = np.array([
            [cam_data["color"]["fx"], 0.0, cam_data["color"]["ppx"]],
            [0.0, cam_data["color"]["fy"], cam_data["color"]["ppy"]],
            [0.0, 0.0, 1.0],
        ])
        # The YAML is for 640×480; scale to actual video resolution
        ref_w, ref_h = 640.0, 480.0
        sx = W_vid / ref_w
        sy = H_vid / ref_h
        K = K_ref.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        print(f"[INFO] Intrinsic matrix (scaled {int(ref_w)}×{int(ref_h)} → {W_vid}×{H_vid})")
    if K is None:
        # Estimate from a ~60° horizontal FOV
        fov_deg = 60.0
        fx = W_vid / (2.0 * np.tan(np.deg2rad(fov_deg / 2.0)))
        fy = fx  # square pixels
        K = np.array([
            [fx,  0.0, W_vid / 2.0],
            [0.0, fy,  H_vid / 2.0],
            [0.0, 0.0, 1.0],
        ])
        print(f"[INFO] Estimated intrinsics from {fov_deg}° FOV")
    print(f"[INFO] K =\n{K}")
    np.savetxt(os.path.join(args.save_dir, "K.txt"), K)

    # ── 3. Extract ALL frames ───────────────────────────────────────────
    print("[INFO] Extracting video frames …")
    all_frames_bgr = []
    for idx, bgr in extract_frames(args.video, skip=args.skip_frames):
        all_frames_bgr.append(bgr)
    all_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_frames_bgr]
    N = len(all_frames_rgb)
    print(f"[INFO] Extracted {N} frames (skip={args.skip_frames})")

    # ── 4. DA3 metric depth ─────────────────────────────────────────────
    print("[INFO] Loading Depth Anything 3 model …")
    da3 = DepthAnything3.from_pretrained(f"depth-anything/{args.da3_model.upper()}")
    da3 = da3.to("cuda")

    print("[INFO] Computing metric depth for all frames …")
    all_depths = []
    da3_intrinsics = None  # Will capture from first batch if use_da3_intrinsics
    bs = args.da3_batch_size
    for start in range(0, N, bs):
        batch = all_frames_rgb[start : start + bs]
        depths, pred = compute_da3_depth(da3, batch)
        # Capture DA3 intrinsics from first batch and scale to video resolution
        if da3_intrinsics is None and pred.intrinsics is not None:
            # DA3 intrinsics are at processing resolution; scale to full video
            proc_h, proc_w = pred.depth.shape[1], pred.depth.shape[2]
            sx_da3 = W_vid / proc_w
            sy_da3 = H_vid / proc_h
            K_da3 = pred.intrinsics[0].copy()  # (3, 3)
            K_da3[0, :] *= sx_da3
            K_da3[1, :] *= sy_da3
            da3_intrinsics = K_da3
        # depths shape: (B, proc_H, proc_W) – resize to video resolution
        for d in depths:
            d_full = resize_depth_to_frame(d, H_vid, W_vid) * args.depth_scale
            all_depths.append(d_full)
        print(f"  DA3 batch {start // bs + 1}/{(N + bs - 1) // bs} done")

    # Optionally use DA3's predicted intrinsics
    if args.use_da3_intrinsics and da3_intrinsics is not None:
        K = da3_intrinsics
        print(f"[INFO] Using DA3 predicted intrinsics (scaled to {W_vid}×{H_vid}):\n{K}")
        np.savetxt(os.path.join(args.save_dir, "K.txt"), K)
    # Free DA3 GPU memory
    del da3
    torch.cuda.empty_cache()
    print(f"[INFO] Depth estimation complete. Depth range frame-0: "
          f"[{all_depths[0].min():.3f}, {all_depths[0].max():.3f}] m")

    # Save first-frame depth for debugging
    depth0_vis = (all_depths[0] / all_depths[0].max() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.save_dir, "depth_frame0.png"), depth0_vis)

    # ── 5. Object segmentation (SAM3 masks or SAM2 fallback) ─────────────
    print("[INFO] Loading object masks …")
    first_rgb = all_frames_rgb[0]
    first_depth = all_depths[0]

    use_sam3 = args.sam3_mask_dir is not None and os.path.isdir(args.sam3_mask_dir)

    # Map track names to GLB paths.  Any track whose name starts with
    # "bottle" (e.g. bottle_0, bottle_1, bottle) uses bottle.glb,
    # any track starting with "scale" uses scale.glb.
    def glb_for_track(track_name: str) -> str | None:
        if track_name.startswith("bottle"):
            return args.bottle_glb
        elif track_name.startswith("scale"):
            return args.scale_glb
        return None

    objects = {}  # name → {"glb": path}
    if use_sam3:
        print(f"[INFO] Using SAM3 masks from: {args.sam3_mask_dir}")
        # Auto-discover tracks by scanning sub-directories
        for entry in sorted(os.listdir(args.sam3_mask_dir)):
            entry_path = os.path.join(args.sam3_mask_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            glb = glb_for_track(entry)
            if glb is None:
                print(f"  {entry}: no GLB mapping, skipping")
                continue
            mask = load_sam3_mask(args.sam3_mask_dir, entry, 0)
            if mask is not None and mask.sum() > 100:
                objects[entry] = {"glb": glb}
                print(f"  {entry}: SAM3 mask loaded ({mask.sum()} pixels) → {glb}")
            else:
                print(f"  {entry}: No valid SAM3 mask for frame 0, skipping")
    else:
        # SAM2 fallback with bounding boxes
        if args.bottle_bbox is not None:
            objects["bottle"] = {"glb": args.bottle_glb, "bbox": np.array(args.bottle_bbox)}
        if args.scale_bbox is not None:
            objects["scale"] = {"glb": args.scale_glb, "bbox": np.array(args.scale_bbox)}

    if len(objects) == 0:
        first_frame_path = os.path.join(args.save_dir, "first_frame.png")
        cv2.imwrite(first_frame_path, all_frames_bgr[0])
        print(f"\n[ACTION REQUIRED] No objects to track.")
        print(f"  Option 1 (recommended): Generate SAM3 masks first:")
        print(f"    conda activate sam3")
        print(f"    python generate_sam3_masks.py --skip_frames {args.skip_frames} --prompts bottle scale")
        print(f"    Then re-run with: --sam3_mask_dir results/sam3_masks")
        print(f"  Option 2: Supply bounding boxes:")
        print(f"    --bottle_bbox x1 y1 x2 y2  --scale_bbox x1 y1 x2 y2")
        return

    # ── 6. For each object: segment → scale mesh → init pose ───────────
    glctx = dr.RasterizeCudaContext()
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    estimators = {}   # name → Any6D instance
    poses = {}        # name → current 4×4 pose
    meshes = {}       # name → scaled trimesh

    for obj_name, obj_info in objects.items():
        print(f"\n{'='*60}")
        print(f"[{obj_name.upper()}] Processing …")

        # 6a. Get mask (SAM3 pre-generated or SAM2 live)
        if use_sam3:
            mask = load_sam3_mask(args.sam3_mask_dir, obj_name, 0)
        else:
            bbox = obj_info["bbox"]
            mask = segment_object_sam2(first_rgb, bbox)
        mask_path = os.path.join(args.save_dir, f"mask_{obj_name}.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        print(f"  Mask saved to {mask_path}  (pixels: {mask.sum()})")

        # 6a2. Downscale for pose estimation to avoid CUDA OOM
        first_rgb_s, first_depth_s, mask_s, K_s, pose_scale = downscale_for_pose(
            first_rgb, first_depth, mask, K, max_side=480,
        )
        print(f"  Pose-estimation resolution: {first_rgb_s.shape[1]}×{first_rgb_s.shape[0]} (scale={pose_scale:.3f})")

        # 6b. Load mesh
        mesh = load_glb_mesh(obj_info["glb"])
        print(f"  Mesh loaded: {mesh.vertices.shape[0]} verts, extents {mesh.extents}")

        # 6c. Create estimator & run register_any6d (includes auto-scaling)
        obj_save = os.path.join(args.save_dir, obj_name)
        os.makedirs(obj_save, exist_ok=True)

        est = Any6D(
            symmetry_tfs=None,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=2,
            debug_dir=obj_save,
        )

        pose = est.register_any6d(
            K=K_s,
            rgb=first_rgb_s,
            depth=first_depth_s,
            ob_mask=mask_s,
            iteration=args.est_refine_iter,
            name=obj_name,
        )
        print(f"  Initial pose:\n{pose}")
        np.savetxt(os.path.join(obj_save, "pose_init.txt"), pose)
        est.mesh.export(os.path.join(obj_save, "scaled_mesh.obj"))

        estimators[obj_name] = est
        poses[obj_name] = pose
        meshes[obj_name] = est.mesh

    # Pre-compute mesh tensors for rendering (avoids recomputing per frame)
    mesh_tensors_cache = {}
    for obj_name, est in estimators.items():
        mesh_tensors_cache[obj_name] = make_mesh_tensors(est.mesh)

    # ── 7. Track across all frames & render overlay ─────────────────────
    print(f"\n[INFO] Tracking {len(estimators)} object(s) across {N} frames …")

    out_fps = fps / args.skip_frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (W_vid, H_vid))  # cv2 expects (W, H)

    for i in range(N):
        rgb = all_frames_rgb[i]
        depth = all_depths[i]

        # Downscale for pose tracking (same as registration)
        rgb_s, depth_s, K_s = downscale_rgb_depth(rgb, depth, K, max_side=480)

        overlay = rgb.copy()

        for obj_name, est in estimators.items():
            if i == 0:
                pose = poses[obj_name]  # already computed
            elif use_sam3:
                # Per-frame registration using SAM3 masks (avoids tracking drift)
                # Use coarse_est=False, refinement=False to skip mesh re-scaling
                # (mesh was already scaled during frame-0 registration)
                mask_i = load_sam3_mask(args.sam3_mask_dir, obj_name, i)
                if mask_i is not None and mask_i.sum() > 100:
                    _, _, mask_i_s, K_s_reg, _ = downscale_for_pose(
                        rgb, depth, mask_i, K, max_side=480,
                    )
                    pose = est.register_any6d(
                        K=K_s_reg,
                        rgb=rgb_s,
                        depth=depth_s,
                        ob_mask=mask_i_s,
                        iteration=args.est_refine_iter,
                        name=f"{obj_name}_f{i}",
                        coarse_est=False,
                        refinement=False,
                    )
                else:
                    # Fallback to tracking if mask missing for this frame
                    pose = est.track_one_any6d(
                        rgb=rgb_s, depth=depth_s, K=K_s,
                        iteration=args.track_refine_iter,
                    )
                poses[obj_name] = pose
            else:
                pose = est.track_one_any6d(
                    rgb=rgb_s,
                    depth=depth_s,
                    K=K_s,
                    iteration=args.track_refine_iter,
                )
                poses[obj_name] = pose

            # Save pose
            obj_save = os.path.join(args.save_dir, obj_name)
            np.savetxt(os.path.join(obj_save, f"pose_{i:05d}.txt"), pose)

            # Render mesh overlay
            overlay = render_mesh_overlay(
                overlay, est.mesh, pose, K, glctx,
                alpha=args.overlay_alpha,
                mesh_tensors=mesh_tensors_cache[obj_name],
            )

            # Draw coordinate axes
            overlay = draw_xyz_axis(
                overlay, ob_in_cam=pose, scale=0.05, K=K,
                thickness=2, transparency=0, is_input_rgb=True,
            )

        # Write frame
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if i % 10 == 0 or i == N - 1:
            print(f"  Frame {i + 1}/{N}")

        # Save a few debug frames
        if i < 5 or i % 50 == 0:
            imageio.imwrite(
                os.path.join(args.save_dir, f"overlay_{i:05d}.png"), overlay
            )

    writer.release()
    print(f"\n[DONE] Output video saved to: {args.output}")
    print(f"       Debug data saved to:  {args.save_dir}")


if __name__ == "__main__":
    main()
