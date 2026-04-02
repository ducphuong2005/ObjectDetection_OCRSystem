
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class TableReconstructor:
    def __init__(self):
        self.paddle_ocr = None
        self._init_ocr()
    
    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="vi",
                # use_gpu=False,
                # show_log=False,
            )
            logger.info("Table OCR engine initialized")
        except ImportError:
            logger.warning("PaddleOCR not available for table reconstruction")
    
    def _parse_paddle_cell(self, result) -> list:
        texts = []
        if not result:
            return texts
        
        data = result
        # Unwrap nested lists
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
        
        if isinstance(data, list):
            for line in data:
                try:
                    if line and isinstance(line, (list, tuple)) and len(line) >= 2:
                        text_conf = line[1]
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            texts.append(str(text_conf[0]))
                    elif isinstance(line, dict):
                        texts.append(str(line.get('text', line.get('rec_text', ''))))
                except (IndexError, TypeError, ValueError):
                    continue
        return texts

    def ocr_cell(self, cell_image: np.ndarray) -> str:

        if cell_image.size == 0 or cell_image.shape[0] < 5 or cell_image.shape[1] < 5:
            return ""
        
        # Method 1: PaddleOCR
        if self.paddle_ocr:
            try:
                result = self.paddle_ocr.ocr(cell_image)
                texts = self._parse_paddle_cell(result)
                if texts:
                    return " ".join(texts).strip()
            except Exception as e:
                logger.debug(f"PaddleOCR cell error: {e}")
        
        # Method 2: Tesseract fallback
        try:
            import pytesseract
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY) if len(cell_image.shape) == 3 else cell_image
            # Threshold for cleaner text
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                thresh, lang="vie+eng", config="--psm 6"
            ).strip()
            return text
        except Exception as e:
            logger.debug(f"Tesseract cell error: {e}")
        
        return ""
    
    def reconstruct(
        self,
        image: np.ndarray,
        cell_grid: List[List[Tuple[int, int, int, int]]],
    ) -> List[List[str]]:
        if not cell_grid:
            logger.warning("Empty cell grid, cannot reconstruct table")
            return []
        
        table_data = []
        
        for row_idx, row in enumerate(cell_grid):
            row_data = []
            for col_idx, (x1, y1, x2, y2) in enumerate(row):
                # Crop cell from image
                cell_img = image[y1:y2, x1:x2]
                
                if cell_img.size == 0:
                    row_data.append("")
                    continue
                
                # OCR the cell
                text = self.ocr_cell(cell_img)
                row_data.append(text)
                
                logger.debug(f"Cell [{row_idx}][{col_idx}]: '{text}'")
            
            table_data.append(row_data)
        
        logger.info(
            f"Reconstructed table: {len(table_data)} rows x "
            f"{len(table_data[0]) if table_data else 0} cols"
        )
        
        return table_data
    
    def reconstruct_with_headers(
        self,
        image: np.ndarray,
        cell_grid: List[List[Tuple[int, int, int, int]]],
    ) -> Dict[str, Any]:
        table_data = self.reconstruct(image, cell_grid)
        
        if not table_data:
            return {"headers": [], "rows": []}
        
        headers = table_data[0]
        rows = table_data[1:]
        
        return {
            "headers": headers,
            "rows": rows,
        }
    
    def to_json_format(
        self, table_data: List[List[str]]
    ) -> List[List[str]]:
        cleaned = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                # Clean whitespace
                clean_text = " ".join(cell.split()) if cell else ""
                cleaned_row.append(clean_text)
            cleaned.append(cleaned_row)
        
        return cleaned
