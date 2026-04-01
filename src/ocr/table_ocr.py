
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

from .note_ocr import parse_paddle_result

logger = logging.getLogger(__name__)


class TableOCR:
    def __init__(self, lang: str = "vi"):
        self.paddle_ocr = None
        self.lang = lang
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
        except ImportError:
            logger.warning("PaddleOCR chưa cài đặt")

    def preprocess_table(self, image: np.ndarray) -> np.ndarray:
        """Phóng to và tăng nét ảnh bảng."""
        h, w = image.shape[:2]
        if max(h, w) < 800:
            scale = 800 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 0.5, sharpened, 0.5, 0)

    def _run_ocr(self, image: np.ndarray) -> List[Dict]:
        """Chạy PaddleOCR và trả kết quả đã parse."""
        if self.paddle_ocr is None:
            return []
        try:
            result = self.paddle_ocr.ocr(image)
        except Exception:
            try:
                result = self.paddle_ocr.predict(image)
            except Exception:
                return []
        return parse_paddle_result(result)

    def _group_by_rows(self, items: List[Dict], threshold: float = None) -> List[List[Dict]]:
        """Nhóm text theo hàng dựa trên vị trí Y, threshold tự động theo chiều cao chữ."""
        if not items:
            return []

        if threshold is None:
            heights = [item["bbox"][3] - item["bbox"][1] for item in items if (item["bbox"][3] - item["bbox"][1]) > 0]
            threshold = max(np.median(heights) * 0.5, 8.0) if heights else 15.0

        sorted_items = sorted(items, key=lambda x: x["center_y"])
        rows = []
        current_row = [sorted_items[0]]
        current_y = sorted_items[0]["center_y"]

        for item in sorted_items[1:]:
            if abs(item["center_y"] - current_y) <= threshold:
                current_row.append(item)
                current_y = np.mean([it["center_y"] for it in current_row])
            else:
                current_row.sort(key=lambda x: x["bbox"][0])
                rows.append(current_row)
                current_row = [item]
                current_y = item["center_y"]

        current_row.sort(key=lambda x: x["bbox"][0])
        rows.append(current_row)
        return rows

    def _align_columns(self, rows: List[List[Dict]]) -> List[List[str]]:
        if not rows:
            return []

        all_left_x = [item["bbox"][0] for row in rows for item in row]
        if not all_left_x:
            return [[item["text"] for item in row] for row in rows]

        # Cluster cạnh trái thành các cột, threshold tự động theo chiều rộng chữ
        all_left_x.sort()
        col_positions = [all_left_x[0]]
        widths = [item["bbox"][2] - item["bbox"][0] for row in rows for item in row if (item["bbox"][2] - item["bbox"][0]) > 0]
        col_thresh = max(np.median(widths) * 0.3, 25.0) if widths else 30.0

        for x in all_left_x[1:]:
            if x - col_positions[-1] > col_thresh:
                col_positions.append(x)
            else:
                col_positions[-1] = (col_positions[-1] + x) / 2

        num_cols = len(col_positions)

        table = []
        for row in rows:
            cells = [""] * num_cols
            for item in row:
                left_x = item["bbox"][0]
                dists = [abs(left_x - cp) for cp in col_positions]
                best_col = dists.index(min(dists))
                if cells[best_col]:
                    cells[best_col] += " " + item["text"]
                else:
                    cells[best_col] = item["text"]
            table.append(cells)
        return table

    def extract(self, image: np.ndarray, return_debug: bool = False) -> Dict[str, Any]:
        """Pipeline OCR bảng đầy đủ."""
        processed = self.preprocess_table(image)
        items = self._run_ocr(processed)

        if not items:
            items = self._run_ocr(image)
        if not items:
            return {"table_data": [], "num_rows": 0, "num_cols": 0}

        rows = self._group_by_rows(items)
        table = self._align_columns(rows)

        return {
            "table_data": table,
            "num_rows": len(table),
            "num_cols": max(len(r) for r in table) if table else 0,
        }
