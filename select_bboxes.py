"""Interactive bounding-box selector for the first frame.

Run this script, draw rectangles around each object, press ENTER to confirm
each selection, and ESC when done.  The coordinates are printed to stdout
in the format expected by run_bottle_video.py.

Usage:
    python select_bboxes.py [--image results/bottle_tracking/first_frame.png]
"""

import cv2
import argparse
import os

boxes = []
drawing = False
ix, iy = -1, -1
current_box = None


def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, current_box, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_show = clone.copy()
            cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select BBoxes", img_show)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        img_show = clone.copy()
        cv2.rectangle(img_show, (current_box[0], current_box[1]),
                      (current_box[2], current_box[3]), (0, 255, 0), 2)
        cv2.imshow("Select BBoxes", img_show)


def main():
    global clone, current_box
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="results/bottle_tracking/first_frame.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        # Extract from video
        cap = cv2.VideoCapture("demo_data/bottle/bottlevid.mp4")
        ret, img = cap.read()
        cap.release()

    # Resize for display if too large
    H, W = img.shape[:2]
    scale = 1.0
    if max(H, W) > 1200:
        scale = 1200.0 / max(H, W)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    clone = img.copy()

    cv2.namedWindow("Select BBoxes")
    cv2.setMouseCallback("Select BBoxes", draw_rect)

    names = ["bottle", "scale"]
    idx = 0

    print(f"\nDraw a box around the '{names[idx]}', then press ENTER.")
    print("Press 's' to skip an object, ESC to finish.\n")

    cv2.imshow("Select BBoxes", img)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 13 or key == 10:  # ENTER
            if current_box is not None:
                # Scale back to original resolution
                bx = [int(c / scale) for c in current_box]
                boxes.append((names[idx], bx))
                print(f"  {names[idx]}: --{names[idx]}_bbox {bx[0]} {bx[1]} {bx[2]} {bx[3]}")
                current_box = None
                idx += 1
                if idx >= len(names):
                    break
                print(f"\nNow draw a box around the '{names[idx]}', then press ENTER.")
        elif key == ord("s"):
            print(f"  Skipping {names[idx]}")
            idx += 1
            if idx >= len(names):
                break
            print(f"\nNow draw a box around the '{names[idx]}', then press ENTER.")

    cv2.destroyAllWindows()

    if boxes:
        print("\n\nFull command:")
        cmd = "python run_bottle_video.py"
        for name, bx in boxes:
            cmd += f" --{name}_bbox {bx[0]} {bx[1]} {bx[2]} {bx[3]}"
        print(cmd)


if __name__ == "__main__":
    main()
