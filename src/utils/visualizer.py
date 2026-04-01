
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Màu cho từng class (BGR)
CLASS_COLORS = {
    "PartDrawing": (255, 255, 0),
    "Note":        (128, 0, 128),
    "Table":       (0, 0, 255),
}
DEFAULT_COLOR = (0, 255, 0)


class Visualizer:
    """Vẽ kết quả phát hiện lên ảnh với bounding box và nhãn."""

    def __init__(self, line_thickness: int = 3, font_scale: float = 0.8, font_thickness: int = 2):
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def draw_detections(self, image: np.ndarray, detections: List[Dict],
                        show_confidence: bool = True, show_id: bool = True) -> np.ndarray:
        """Vẽ tất cả detection lên ảnh."""
        vis_image = image.copy()

        for det in detections:
            cls_name = det["class"]
            bbox = det["bbox"]
            confidence = det.get("confidence", 0)
            det_id = det.get("id", 0)
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, self.line_thickness)

            label_parts = []
            if show_id:
                label_parts.append(f"#{det_id}")
            label_parts.append(cls_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            label = " ".join(label_parts)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
            label_y = max(y1 - 10, th + 10)
            cv2.rectangle(vis_image, (x1, label_y - th - 5), (x1 + tw + 10, label_y + 5), color, -1)
            cv2.putText(vis_image, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, (255, 255, 255), self.font_thickness)

        return vis_image

    def draw_table_grid(self, image: np.ndarray, cells: List[List[Tuple[int, int, int, int]]],
                        color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Vẽ lưới ô bảng lên ảnh."""
        vis = image.copy()
        for row in cells:
            for (x1, y1, x2, y2) in row:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        return vis

    def create_summary(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Tạo ảnh tổng hợp với thống kê detection ở thanh dưới."""
        vis = self.draw_detections(image, detections)

        counts = {}
        for det in detections:
            cls = det["class"]
            counts[cls] = counts.get(cls, 0) + 1

        h, w = vis.shape[:2]
        bar = np.zeros((40, w, 3), dtype=np.uint8)
        bar[:] = (50, 50, 50)

        stats_text = " | ".join(f"{cls}: {cnt}" for cls, cnt in sorted(counts.items()))
        stats_text = f"Tổng: {len(detections)} | {stats_text}"
        cv2.putText(bar, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return np.vstack([vis, bar])

    def save(self, image: np.ndarray, output_path: str):
        """Lưu ảnh kết quả."""
        cv2.imwrite(output_path, image)
