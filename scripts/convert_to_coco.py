
import os
import json
import glob
import random
import argparse
from datetime import datetime
from typing import Dict, List

import cv2


def load_annotations(annotations_dir: str) -> Dict[str, List[Dict]]:
    """Đọc tất cả file annotation từng ảnh."""
    all_anns = {}
    for ann_file in sorted(glob.glob(os.path.join(annotations_dir, "*.json"))):
        base_name = os.path.splitext(os.path.basename(ann_file))[0]
        with open(ann_file, "r") as f:
            all_anns[base_name] = json.load(f)
    return all_anns


def find_image_file(images_dir: str, base_name: str) -> str:
    """Tìm file ảnh thực tế cho base_name."""
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
        path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(path):
            return base_name + ext
    return None


def build_coco_dataset(annotations: Dict[str, List[Dict]], images_dir: str, image_names: List[str]) -> Dict:
    """Tạo dict format COCO từ annotations."""
    categories = [
        {"id": 0, "name": "PartDrawing"},
        {"id": 1, "name": "Note"},
        {"id": 2, "name": "Table"},
    ]

    images, coco_anns = [], []
    ann_id = 1

    for img_id, base_name in enumerate(image_names, start=1):
        image_file = find_image_file(images_dir, base_name)
        if image_file is None:
            print(f"Cảnh báo: Không tìm thấy ảnh cho {base_name}")
            continue

        img = cv2.imread(os.path.join(images_dir, image_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        images.append({"id": img_id, "file_name": image_file, "width": w, "height": h})

        for ann in annotations.get(base_name, []):
            bbox = ann["bbox"]
            coco_anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann.get("class_id", 0),
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    return {
        "info": {"description": "BOM Object Detection Dataset", "version": "1.0",
                 "year": datetime.now().year, "date_created": datetime.now().isoformat()},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": coco_anns,
    }


def main():
    parser = argparse.ArgumentParser(description="Chuyển annotation sang COCO format")
    parser.add_argument("--annotations-dir", type=str, default="data/annotations")
    parser.add_argument("--images-dir", type=str, default="data/raw/BOM-Dataset")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotations = load_annotations(args.annotations_dir)
    print(f"Đã đọc annotation cho {len(annotations)} ảnh")

    if not annotations:
        print("Không có annotation! Chạy annotate.py trước.")
        return

    all_names = sorted(annotations.keys())
    random.seed(args.seed)
    random.shuffle(all_names)

    val_count = max(1, int(len(all_names) * args.val_ratio))
    val_names = all_names[:val_count]
    train_names = all_names[val_count:]
    print(f"Split: {len(train_names)} train, {len(val_names)} val")

    class_counts = {"PartDrawing": 0, "Note": 0, "Table": 0}
    for anns in annotations.values():
        for ann in anns:
            cls = ann.get("class", "Unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"Phân bố class: {class_counts}")

    for split_name, split_names in [("train", train_names), ("val", val_names)]:
        coco_data = build_coco_dataset(annotations, args.images_dir, split_names)
        path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(coco_data, f, indent=2)
        print(f"Lưu {split_name}: {path} ({len(coco_data['images'])} ảnh, {len(coco_data['annotations'])} annotations)")


if __name__ == "__main__":
    main()
