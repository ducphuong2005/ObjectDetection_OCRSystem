
import cv2
import numpy as np
import logging
from typing import Tuple

from .note_ocr import parse_paddle_result

logger = logging.getLogger(__name__)


class OCREnsemble:
    """Kết hợp PaddleOCR và Tesseract, chọn kết quả confidence cao hơn."""

    def __init__(self, lang: str = "vi"):
        self.lang = lang
        self.paddle_ocr = None
        self.tesseract_available = False
        self._init_engines()

    def _init_engines(self):
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
        except ImportError:
            pass
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
        except Exception:
            pass

    def ocr_paddle(self, image: np.ndarray) -> Tuple[str, float]:
        if not self.paddle_ocr:
            return "", 0.0
        try:
            result = self.paddle_ocr.ocr(image)
            items = parse_paddle_result(result)
            if not items:
                return "", 0.0
            items.sort(key=lambda x: x["center_y"])
            texts = [item["text"] for item in items]
            confs = [item["confidence"] for item in items]
            return "\n".join(texts), float(np.mean(confs))
        except Exception:
            return "", 0.0

    def ocr_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        if not self.tesseract_available:
            return "", 0.0
        try:
            import pytesseract
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            data = pytesseract.image_to_data(binary, lang="eng", output_type=pytesseract.Output.DICT)
            texts, confs = [], []
            for i, text in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > 0 and text.strip():
                    texts.append(text)
                    confs.append(conf / 100.0)
            return " ".join(texts), float(np.mean(confs)) if confs else 0.0
        except Exception:
            return "", 0.0

    def extract(self, image: np.ndarray) -> str:
        paddle_text, paddle_conf = self.ocr_paddle(image)
        tess_text, tess_conf = self.ocr_tesseract(image)
        if not paddle_text.strip():
            return tess_text
        if not tess_text.strip():
            return paddle_text
        return paddle_text if paddle_conf >= tess_conf else tess_text
