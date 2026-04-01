
import os
import json
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..detection.inference import ObjectDetector
from ..utils.cropper import ObjectCropper
from ..ocr.note_ocr import NoteOCR
from ..ocr.table_ocr import TableOCR
from ..utils.postprocess import OCRPostProcessor
from ..utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


class BOMPipeline:
    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold

        logger.info("Khởi tạo BOM Pipeline...")
        self.detector = ObjectDetector(model_path=model_path, confidence_threshold=confidence_threshold)
        self.cropper = ObjectCropper(padding=10, output_dir=os.path.join(output_dir, "crops"))
        self.note_ocr = NoteOCR(lang="vi")
        self.table_ocr = TableOCR()
        self.postprocessor = OCRPostProcessor()
        self.visualizer = Visualizer()

        for subdir in ["json", "crops", "visualizations"]:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        logger.info("BOM Pipeline khởi tạo thành công")

    def process(self, image_path: str, save_outputs: bool = True) -> Dict[str, Any]:
        """Xử lý 1 ảnh qua toàn bộ pipeline."""
        image_name = os.path.basename(image_path)
        logger.info(f"Đang xử lý: {image_name}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        # Bước 1: Phát hiện đối tượng
        detections = self.detector.detect(image)
        if not detections:
            return self._empty_result(image_name)

        # Bước 2: Cắt vùng
        crops = self.cropper.crop_objects(image, detections, image_name, save=save_outputs)

        # Bước 3: OCR từng vùng
        objects = [self._process_single_object(crop_info) for crop_info in crops]

        # Bước 4: Xuất JSON
        result = {
            "image": image_name,
            "timestamp": datetime.now().isoformat(),
            "num_objects": len(objects),
            "objects": objects,
        }

        # Bước 5: Trực quan hoá
        vis_image = self.visualizer.create_summary(image, detections)
        if save_outputs:
            self._save_outputs(result, vis_image, image_name)

        logger.info(f"Hoàn tất {image_name}: {len(objects)} đối tượng")
        return result

    def _process_single_object(self, crop_info: Dict) -> Dict[str, Any]:
        """OCR cho 1 đối tượng dựa theo class."""
        cls_name = crop_info["class"]
        crop_image = crop_info["crop"]
        ocr_content = None

        if cls_name == "Note":
            raw_text = self.note_ocr.extract(crop_image, use_ensemble=True)
            ocr_content = self.postprocessor.process(raw_text, context="note")
        elif cls_name == "Table":
            table_result = self.table_ocr.extract(crop_image)
            table_data = table_result.get("table_data", [])
            ocr_content = self.postprocessor.process_table(table_data)

        return {
            "id": crop_info["id"],
            "class": cls_name,
            "confidence": crop_info["confidence"],
            "bbox": crop_info["bbox"],
            "ocr_content": ocr_content,
            "crop_path": crop_info.get("crop_path"),
        }

    def _empty_result(self, image_name: str) -> Dict[str, Any]:
        return {"image": image_name, "timestamp": datetime.now().isoformat(), "num_objects": 0, "objects": []}

    def _save_outputs(self, result: Dict, vis_image: np.ndarray, image_name: str):
        """Lưu JSON và ảnh trực quan."""
        base_name = os.path.splitext(image_name)[0]

        json_path = os.path.join(self.output_dir, "json", f"{base_name}.json")
        save_result = self._make_serializable(result)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2)

        vis_path = os.path.join(self.output_dir, "visualizations", f"{base_name}_vis.jpg")
        cv2.imwrite(vis_path, vis_image)

    def _make_serializable(self, obj: Any) -> Any:
        """Loại bỏ numpy array để JSON serialize được."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() if k != "crop"}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    def process_batch(self, image_dir: str, save_outputs: bool = True) -> List[Dict]:
        """Xử lý tất cả ảnh trong thư mục."""
        supported_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        image_files = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in supported_exts])

        logger.info(f"Xử lý {len(image_files)} ảnh từ {image_dir}")
        results = []
        for i, filename in enumerate(image_files):
            image_path = os.path.join(image_dir, filename)
            logger.info(f"[{i+1}/{len(image_files)}] {filename}")
            try:
                results.append(self.process(image_path, save_outputs=save_outputs))
            except Exception as e:
                logger.error(f"Lỗi xử lý {filename}: {e}")
                results.append(self._empty_result(filename))

        if save_outputs:
            summary_path = os.path.join(self.output_dir, "json", "batch_results.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump([self._make_serializable(r) for r in results], f, ensure_ascii=False, indent=2)

        return results
