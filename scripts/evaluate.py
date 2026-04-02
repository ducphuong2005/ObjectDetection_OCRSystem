
import os
import sys
import json
import argparse
import cv2
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.inference import ObjectDetector
from src.utils.visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, images_dir: str,
                   output_dir: str = "outputs/evaluation", confidence_threshold: float = 0.5):
    """Đánh giá model detection trên tập ảnh."""
    os.makedirs(output_dir, exist_ok=True)

    detector = ObjectDetector(model_path=model_path, confidence_threshold=confidence_threshold)
    visualizer = Visualizer()

    supported_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = sorted([f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in supported_exts])

    print(f"Đánh giá {len(image_files)} ảnh...")

    all_results = []
    class_counts = {}

    for i, filename in enumerate(image_files):
        image = cv2.imread(os.path.join(images_dir, filename))
        if image is None:
            continue

        detections = detector.detect(image)
        for det in detections:
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        vis = visualizer.create_summary(image, detections)
        cv2.imwrite(os.path.join(output_dir, f"eval_{filename}"), vis)

        all_results.append({"image": filename, "num_detections": len(detections), "detections": detections})
        print(f"  [{i+1}/{len(image_files)}] {filename}: {len(detections)} phát hiện")

    total = sum(class_counts.values())
    print(f"\n=== Tổng kết đánh giá ===")
    print(f"Tổng ảnh: {len(image_files)}")
    print(f"Tổng phát hiện: {total}")
    print(f"Phân bố class:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    print(f"Trung bình/ảnh: {total / max(len(image_files), 1):.1f}")

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "summary": {"total_images": len(image_files), "total_detections": total,
                        "class_counts": class_counts, "confidence_threshold": confidence_threshold},
            "results": all_results,
        }, f, indent=2)
    print(f"\nKết quả lưu tại {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá BOM Detection Model")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn model weights")
    parser.add_argument("--images", type=str, default="data/raw/BOM-Dataset")
    parser.add_argument("--output", type=str, default="outputs/evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_model(args.model, args.images, args.output, args.threshold)
