
import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ObjectCropper:
    """Cắt vùng đối tượng phát hiện được với padding tuỳ chỉnh."""

    def __init__(self, padding: int = 10, output_dir: str = "outputs/crops"):
        self.padding = padding
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def crop_single(self, image: np.ndarray, bbox: Dict[str, float], padding: Optional[int] = None) -> np.ndarray:
        """Cắt 1 vùng từ ảnh theo bbox {x1, y1, x2, y2}."""
        pad = padding if padding is not None else self.padding
        h, w = image.shape[:2]

        x1 = max(0, int(bbox["x1"]) - pad)
        y1 = max(0, int(bbox["y1"]) - pad)
        x2 = min(w, int(bbox["x2"]) + pad)
        y2 = min(h, int(bbox["y2"]) + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning(f"Crop rỗng cho bbox {bbox}")
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return crop

    def crop_objects(self, image: np.ndarray, detections: List[Dict],
                     image_name: str = "image", save: bool = True) -> List[Dict]:
        """Cắt tất cả đối tượng phát hiện được, lưu vào thư mục theo class."""
        results = []
        class_counters = {}

        for det in detections:
            cls = det["class"]
            class_counters[cls] = class_counters.get(cls, 0) + 1
            crop = self.crop_single(image, det["bbox"])

            clean_name = os.path.splitext(image_name)[0]
            filename = f"{clean_name}_{cls.lower()}_{class_counters[cls]}.png"

            save_path = None
            if save:
                cls_dir = os.path.join(self.output_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                save_path = os.path.join(cls_dir, filename)
                cv2.imwrite(save_path, crop)

            results.append({**det, "crop": crop, "crop_path": save_path})

        logger.info(f"Đã cắt {len(results)} đối tượng từ {image_name}")
        return results
