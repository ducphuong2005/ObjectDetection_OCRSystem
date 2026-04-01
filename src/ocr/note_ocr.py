
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def parse_paddle_result(result) -> List[Dict[str, Any]]:

    items = []
    if result is None:
        return items

    # === PaddleOCR v5: dict hoặc list chứa dict ===
    if isinstance(result, dict):
        return _parse_v5_dict(result)

    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict):
            return _parse_v5_dict(result[0])
        first = result[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
            return _parse_v5_dict(first[0])

    # === PaddleOCR v4: nested lists ===
    return _parse_v4_list(result)


def _parse_v5_dict(data: dict) -> List[Dict[str, Any]]:
    """Parse format v5: dict có rec_texts, rec_scores, dt_polys."""
    items = []
    rec_texts = data.get('rec_texts', [])
    rec_scores = data.get('rec_scores', [])
    dt_polys = data.get('dt_polys', [])

    for i, text in enumerate(rec_texts):
        text = str(text).strip()
        if not text:
            continue
        conf = float(rec_scores[i]) if i < len(rec_scores) else 0.5

        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        if i < len(dt_polys):
            poly = np.array(dt_polys[i])
            if poly.ndim >= 2 and poly.shape[0] >= 4:
                x1 = float(np.min(poly[:, 0]))
                y1 = float(np.min(poly[:, 1]))
                x2 = float(np.max(poly[:, 0]))
                y2 = float(np.max(poly[:, 1]))

        items.append({
            "text": text, "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
        })
    return items


def _parse_v4_list(result) -> List[Dict[str, Any]]:
    """Parse format v4: nested list [[bbox, (text, conf)], ...]."""
    items = []
    data = result
    while isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            if isinstance(first[1], (list, tuple)) and len(first[1]) >= 2:
                if isinstance(first[1][0], str):
                    break
        if isinstance(first, list):
            data = first
        else:
            break

    if not isinstance(data, list):
        return items

    for item in data:
        try:
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                bbox_points = item[0]
                text_conf = item[1]

                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    text = str(text_conf[0]).strip()
                    conf = float(text_conf[1])
                elif isinstance(text_conf, str):
                    text = text_conf.strip()
                    conf = 0.5
                else:
                    continue
                if not text:
                    continue

                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                if isinstance(bbox_points, (list, np.ndarray)) and len(bbox_points) >= 4:
                    points = np.array(bbox_points)
                    x1 = float(np.min(points[:, 0]))
                    y1 = float(np.min(points[:, 1]))
                    x2 = float(np.max(points[:, 0]))
                    y2 = float(np.max(points[:, 1]))

                items.append({
                    "text": text, "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                })
        except (IndexError, TypeError, ValueError):
            continue

    return items


class NoteOCR:
    """Engine OCR cho vùng Note trong bản vẽ kỹ thuật."""

    def __init__(self, lang: str = "vi", use_gpu: bool = False):
        self.lang = lang
        self.paddle_ocr = None
        self.tesseract_available = False
        self._init_paddle()
        self._init_tesseract()

    def _init_paddle(self):
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
        except ImportError:
            logger.warning("PaddleOCR chưa cài đặt")

    def _init_tesseract(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
        except Exception:
            pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý nhẹ: phóng to ảnh nhỏ và tăng nét."""
        h, w = image.shape[:2]
        if max(h, w) < 600:
            scale = 600 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def extract_paddle(self, image: np.ndarray) -> Tuple[str, float]:
        """Trích xuất text bằng PaddleOCR."""
        if self.paddle_ocr is None:
            return "", 0.0
        try:
            result = self.paddle_ocr.ocr(image)
        except Exception:
            try:
                result = self.paddle_ocr.predict(image)
            except Exception:
                return "", 0.0

        items = parse_paddle_result(result)
        if not items:
            return "", 0.0

        items.sort(key=lambda x: x["center_y"])
        texts = [item["text"] for item in items]
        confs = [item["confidence"] for item in items]
        return "\n".join(texts), float(np.mean(confs))

    def extract_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        """Trích xuất text bằng Tesseract (fallback)."""
        if not self.tesseract_available:
            return "", 0.0
        try:
            import pytesseract
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            data = pytesseract.image_to_data(binary, lang="eng", output_type=pytesseract.Output.DICT)

            texts, confs = [], []
            current_line, current_num = [], -1
            for i, text in enumerate(data["text"]):
                conf = int(data["conf"][i])
                line_num = data["line_num"][i]
                if conf > 0 and text.strip():
                    if line_num != current_num and current_line:
                        texts.append(" ".join(current_line))
                        current_line = []
                    current_line.append(text)
                    current_num = line_num
                    confs.append(conf / 100.0)
            if current_line:
                texts.append(" ".join(current_line))
            return "\n".join(texts), float(np.mean(confs)) if confs else 0.0
        except Exception:
            return "", 0.0

    def extract(self, image: np.ndarray, use_ensemble: bool = True) -> str:
        """
        Trích xuất text từ vùng Note.
        Thử: ảnh đã xử lý → ảnh gốc → Tesseract fallback.
        """
        preprocessed = self.preprocess(image)
        text1, conf1 = self.extract_paddle(preprocessed)
        text2, conf2 = self.extract_paddle(image)

        if conf1 >= conf2:
            paddle_text, paddle_conf = text1, conf1
        else:
            paddle_text, paddle_conf = text2, conf2

        if not use_ensemble or not self.tesseract_available:
            return paddle_text

        tess_text, tess_conf = self.extract_tesseract(image)
        return paddle_text if paddle_conf >= tess_conf else tess_text
