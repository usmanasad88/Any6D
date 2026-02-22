"""
Fair comparison: run PoseTrackingMonitor on the same video/data as run_bottle_video.py.

Run in the Any6D conda env:
    conda activate Any6D
    python test_monitor_comparison.py

Compares output to: results/bottle_tracked_sam3.mp4
"""
from __future__ import annotations

import os
import sys

# ── Make aura importable from the Any6D conda env ─────────────────────────
AURA_SRC = os.path.expanduser("~/Repos/aura/src")
if AURA_SRC not in sys.path:
    sys.path.insert(0, AURA_SRC)

# ── Paths (must match the original run_bottle_video.py invocation) ────────
ANY6D_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.expanduser(
    "~/Repos/aura/demo_data/layup_demo/mesh3d/bottlevid.mp4"
)
BOTTLE_GLB = os.path.expanduser(
    "~/Repos/ur_ws/isaac_standalone/Objects/bottle.glb"
)
SCALE_GLB = os.path.expanduser(
    "~/Repos/ur_ws/isaac_standalone/Objects/scale.glb"
)
SAM3_MASK_DIR = os.path.join(ANY6D_ROOT, "results", "sam3_masks")
INTRINSIC_FILE = os.path.join(ANY6D_ROOT, "demo_data", "836212060125_640x480.yml")

OUTPUT_VIDEO = os.path.join(ANY6D_ROOT, "results", "monitor_comparison.mp4")
SAVE_DIR = os.path.join(ANY6D_ROOT, "results", "monitor_comparison")

SKIP_FRAMES = 5


def main():
    # Verify all inputs exist
    for label, path in [
        ("Video", VIDEO_PATH),
        ("Bottle GLB", BOTTLE_GLB),
        ("Scale GLB", SCALE_GLB),
        ("SAM3 masks", SAM3_MASK_DIR),
        ("Intrinsics", INTRINSIC_FILE),
    ]:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if not exists:
            print(f"\nERROR: {label} not found. Cannot proceed.")
            return

    # ── Build config matching original run_bottle_video.py defaults ───────
    from aura.utils.config import PoseTrackingConfig

    config = PoseTrackingConfig(
        # Mesh mapping: prefix → GLB path
        # Tracks "bottle_0", "bottle_1" start with "bottle" → bottle.glb
        # Track "scale" starts with "scale" → scale.glb
        mesh_map={
            "bottle": BOTTLE_GLB,
            "scale": SCALE_GLB,
        },
        # Pre-generated SAM3 masks (same as original --sam3_mask_dir)
        sam3_mask_dir=SAM3_MASK_DIR,
        # Intrinsics (same as original --intrinsic_file)
        intrinsic_file=INTRINSIC_FILE,
        # DA3 model (same as original --da3_model)
        da3_model="da3nested-giant-large",
        da3_batch_size=8,
        depth_scale=1.0,
        # Use DA3 predicted intrinsics (same as original --use_da3_intrinsics)
        use_da3_intrinsics=True,
        # Pose estimation (same as original defaults)
        est_refine_iter=5,
        track_refine_iter=2,
        max_pose_resolution=480,
        # Rendering (same as original --overlay_alpha 0.6)
        overlay_alpha=0.6,
        render_overlay=True,
        render_axes=True,
        # Debug level 2 to match original debug=2
        debug_level=2,
        save_dir=SAVE_DIR,
        # Point to this repo as the Any6D root
        any6d_root=ANY6D_ROOT,
    )

    print(f"\n[CONFIG]")
    print(f"  da3_model:          {config.da3_model}")
    print(f"  use_da3_intrinsics: {config.use_da3_intrinsics}")
    print(f"  est_refine_iter:    {config.est_refine_iter}")
    print(f"  track_refine_iter:  {config.track_refine_iter}")
    print(f"  overlay_alpha:      {config.overlay_alpha}")
    print(f"  debug_level:        {config.debug_level}")
    print(f"  sam3_mask_dir:      {config.sam3_mask_dir}")
    print(f"  skip_frames:        {SKIP_FRAMES}")
    print(f"  output:             {OUTPUT_VIDEO}")

    # ── Create and run the monitor ────────────────────────────────────────
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from aura.monitors.pose_tracking_monitor import PoseTrackingMonitor

    monitor = PoseTrackingMonitor(config)

    print(f"\n[RUN] Processing video through PoseTrackingMonitor …")
    output_path = monitor.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_VIDEO,
        skip_frames=SKIP_FRAMES,
    )

    print(f"\n{'='*60}")
    print(f"[DONE] Monitor output:   {output_path}")
    print(f"       Debug artifacts:  {SAVE_DIR}")
    print(f"\nCompare with original:  results/bottle_tracked_sam3.mp4")
    print(f"  e.g.  ffplay {output_path}")
    print(f"        ffplay results/bottle_tracked_sam3.mp4")


if __name__ == "__main__":
    main()
