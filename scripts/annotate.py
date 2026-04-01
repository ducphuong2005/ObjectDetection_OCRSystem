
import os
import json
import cv2
import glob
import argparse
import numpy as np
from typing import Dict, List

CLASSES = {1: "PartDrawing", 2: "Note", 3: "Table"}
CLASS_COLORS = {"PartDrawing": (255, 255, 0), "Note": (128, 0, 128), "Table": (0, 0, 255)}

current_class = 1
drawing = False
ix, iy = 0, 0
annotations = []
temp_rect = None


def draw_callback(event, x, y, flags, param):
    """Callback chuột để vẽ bounding box."""
    global drawing, ix, iy, temp_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_rect = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_rect = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - ix) > 10 and abs(y - iy) > 10:
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            ann = {"bbox": [x1, y1, x2 - x1, y2 - y1], "class": CLASSES[current_class], "class_id": current_class - 1}
            annotations.append(ann)
            print(f"  Thêm: {ann['class']} tại [{x1}, {y1}, {x2-x1}, {y2-y1}]")
        temp_rect = None


def render(image, anns, temp):
    """Vẽ annotation hiện tại lên ảnh."""
    vis = image.copy()
    for ann in anns:
        x, y, w, h = ann["bbox"]
        color = CLASS_COLORS.get(ann["class"], (0, 255, 0))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, ann["class"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if temp:
        x1, y1, x2, y2 = temp
        color = CLASS_COLORS.get(CLASSES[current_class], (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

    h, w = vis.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    status = f"Class: {CLASSES[current_class]} (1/2/3) | Annotations: {len(anns)} | [u]ndo [n]ext [p]rev [s]ave [q]uit"
    cv2.putText(bar, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([vis, bar])


def save_annotations(output_dir: str, base_name: str, anns: List[Dict]):
    """Lưu annotation ra file JSON."""
    path = os.path.join(output_dir, f"{base_name}.json")
    with open(path, "w") as f:
        json.dump(anns, f, indent=2)
    print(f"  Đã lưu {len(anns)} annotations → {path}")


def main():
    global current_class, annotations, temp_rect

    parser = argparse.ArgumentParser(description="Công cụ gán nhãn BOM")
    parser.add_argument("--data-dir", type=str, default="data/raw/BOM-Dataset")
    parser.add_argument("--output-dir", type=str, default="data/annotations")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = sorted([f for f in glob.glob(os.path.join(args.data_dir, "*"))
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))])

    if not image_files:
        print(f"Không tìm thấy ảnh trong {args.data_dir}")
        return

    print(f"Tìm thấy {len(image_files)} ảnh")
    print("Điều khiển: 1=PartDrawing, 2=Note, 3=Table, u=undo, n=next, p=prev, s=save, q=quit")

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotator", draw_callback)
    idx = args.start_index

    while 0 <= idx < len(image_files):
        image_path = image_files[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        if image is None:
            idx += 1
            continue

        ann_path = os.path.join(args.output_dir, f"{base_name}.json")
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                annotations = json.load(f)
            print(f"\n[{idx+1}/{len(image_files)}] {os.path.basename(image_path)} (đã có {len(annotations)} annotations)")
        else:
            annotations = []
            print(f"\n[{idx+1}/{len(image_files)}] {os.path.basename(image_path)} (mới)")

        while True:
            display = render(image, annotations, temp_rect)
            cv2.imshow("Annotator", display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('1'):
                current_class = 1
            elif key == ord('2'):
                current_class = 2
            elif key == ord('3'):
                current_class = 3
            elif key == ord('u') and annotations:
                removed = annotations.pop()
                print(f"  Hoàn tác: {removed['class']}")
            elif key == ord('s'):
                save_annotations(args.output_dir, base_name, annotations)
            elif key == ord('n'):
                save_annotations(args.output_dir, base_name, annotations)
                idx += 1
                break
            elif key == ord('p'):
                save_annotations(args.output_dir, base_name, annotations)
                idx -= 1
                break
            elif key == ord('q'):
                save_annotations(args.output_dir, base_name, annotations)
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\nHoàn tất gán nhãn!")


if __name__ == "__main__":
    main()
