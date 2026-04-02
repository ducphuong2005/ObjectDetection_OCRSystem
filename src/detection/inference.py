
import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Phát hiện đối tượng sử dụng Detectron2 Faster R-CNN."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.predictor = None
        self.cfg = None
        self._setup(device)

    def _setup(self, device: str):
        """Khởi tạo Detectron2 predictor."""
        try:
            import torch
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from .config import DETECTION_CONFIG, CLASSES

            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file(DETECTION_CONFIG["model_zoo_config"])
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = DETECTION_CONFIG["num_classes"]
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold

            # Tìm model weights
            if self.model_path and os.path.exists(self.model_path):
                cfg.MODEL.WEIGHTS = self.model_path
            else:
                default_path = os.path.join(
                    DETECTION_CONFIG["output_dir"], "model_final.pth"
                )
                if os.path.exists(default_path):
                    cfg.MODEL.WEIGHTS = default_path
                else:
                    logger.warning(
                        f"Không tìm thấy model tại {default_path}. "
                        "Sử dụng pretrained COCO weights."
                    )
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                        DETECTION_CONFIG["model_zoo_config"]
                    )

            if device == "auto":
                cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                cfg.MODEL.DEVICE = device

            self.cfg = cfg
            self.predictor = DefaultPredictor(cfg)
            self.classes = CLASSES

            logger.info(f"Detector khởi tạo trên {cfg.MODEL.DEVICE}")
            logger.info(f"Model: {cfg.MODEL.WEIGHTS}")

        except ImportError:
            logger.warning("Detectron2 chưa cài. Dùng chế độ fallback.")
            self.predictor = None
            self.classes = ["PartDrawing", "Note", "Table"]

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Phát hiện đối tượng trên ảnh.

        Args:
            image: Ảnh BGR (OpenCV format)

        Returns:
            Danh sách dict: class, class_id, confidence, bbox {x1, y1, x2, y2}
        """
        if self.predictor is None:
            return []

        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        detections = []
        for i in range(len(boxes)):
            score = float(scores[i])
            if score < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = boxes[i].tolist()
            class_id = int(classes[i])
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"

            detections.append({
                "id": i + 1,
                "class": class_name,
                "class_id": class_id,
                "confidence": round(score, 4),
                "bbox": {
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                },
            })

        # Sắp xếp theo confidence giảm dần
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        for idx, det in enumerate(detections):
            det["id"] = idx + 1

        logger.info(f"Phát hiện {len(detections)} đối tượng: {', '.join(d['class'] for d in detections)}")
        return detections

    def detect_from_file(self, image_path: str) -> List[Dict]:
        """Phát hiện đối tượng từ file ảnh."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
        return self.detect(image)
