
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class CellDetector:

    def __init__(
        self,
        min_line_length: int = 30,
        kernel_scale: int = 15,
    ):

        self.min_line_length = min_line_length
        self.kernel_scale = kernel_scale
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Adaptive threshold to handle varying illumination
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=9,
        )
        
        return binary
    
    def detect_lines(
        self, binary: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        h, w = binary.shape[:2]
        
        # Horizontal lines
        h_kernel_size = max(w // self.kernel_scale, self.min_line_length)
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (h_kernel_size, 1)
        )
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        # Vertical lines
        v_kernel_size = max(h // self.kernel_scale, self.min_line_length)
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, v_kernel_size)
        )
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
        
        # Dilate to connect broken lines
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        horizontal = cv2.dilate(horizontal, dilate_kernel, iterations=1)
        vertical = cv2.dilate(vertical, dilate_kernel, iterations=1)
        
        return horizontal, vertical
    
    def find_intersections(
        self,
        horizontal: np.ndarray,
        vertical: np.ndarray,
    ) -> np.ndarray:
        intersections = cv2.bitwise_and(horizontal, vertical)
        
        # Dilate intersections to merge nearby points
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        intersections = cv2.dilate(intersections, kernel, iterations=2)
        
        return intersections
    
    def extract_grid_points(
        self, intersections: np.ndarray
    ) -> Tuple[List[int], List[int]]:

        # Find contours of intersection regions
        contours, _ = cv2.findContours(
            intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return [], []
        
        # Get center points of intersection regions
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))
        
        if not points:
            return [], []
        
        # Cluster points into rows and columns
        xs = sorted(set(p[0] for p in points))
        ys = sorted(set(p[1] for p in points))
        
        # Merge close points (within threshold)
        def merge_close(values: List[int], threshold: int = 15) -> List[int]:
            if not values:
                return []
            merged = [values[0]]
            for v in values[1:]:
                if v - merged[-1] > threshold:
                    merged.append(v)
                else:
                    # Average with previous
                    merged[-1] = (merged[-1] + v) // 2
            return merged
        
        cols = merge_close(xs)
        rows = merge_close(ys)
        
        logger.info(f"Grid: {len(rows)} rows x {len(cols)} cols")
        
        return rows, cols
    
    def build_cell_grid(
        self,
        rows: List[int],
        cols: List[int],
        image_shape: Tuple[int, int],
    ) -> List[List[Tuple[int, int, int, int]]]:

        if len(rows) < 2 or len(cols) < 2:
            logger.warning("Not enough grid lines to form cells")
            return []
        
        cells = []
        for i in range(len(rows) - 1):
            row_cells = []
            for j in range(len(cols) - 1):
                x1 = cols[j]
                y1 = rows[i]
                x2 = cols[j + 1]
                y2 = rows[i + 1]
                
                # Add small padding inward to avoid line artifacts
                pad = 3
                x1 = min(x1 + pad, x2)
                y1 = min(y1 + pad, y2)
                x2 = max(x2 - pad, x1)
                y2 = max(y2 - pad, y1)
                
                row_cells.append((x1, y1, x2, y2))
            cells.append(row_cells)
        
        logger.info(f"Built {len(cells)} x {len(cells[0]) if cells else 0} cell grid")
        
        return cells
    
    def detect_cells(
        self, image: np.ndarray
    ) -> Tuple[List[List[Tuple[int, int, int, int]]], np.ndarray]:

        binary = self.preprocess(image)
        horizontal, vertical = self.detect_lines(binary)
        intersections = self.find_intersections(horizontal, vertical)
        rows, cols = self.extract_grid_points(intersections)
        
        h, w = image.shape[:2]
        cells = self.build_cell_grid(rows, cols, (h, w))
        
        # Create debug visualization
        debug = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw detected lines
        line_mask = cv2.bitwise_or(horizontal, vertical)
        debug[line_mask > 0] = [0, 0, 255]  # Red for lines
        
        # Draw cell boundaries
        for row in cells:
            for (x1, y1, x2, y2) in row:
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        return cells, debug
    
    def detect_cells_fallback(
        self, image: np.ndarray
    ) -> List[List[Tuple[int, int, int, int]]]:

        binary = self.preprocess(image)
        
        # Find all contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter for rectangular regions (potential cells)
        cell_rects = []
        h, w = image.shape[:2]
        min_area = (w * h) * 0.001  # Min 0.1% of image area
        max_area = (w * h) * 0.5    # Max 50% of image area
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Check for rectangular aspect
                aspect = cw / max(ch, 1)
                if 0.1 < aspect < 20:
                    cell_rects.append((x, y, x + cw, y + ch))
        
        if not cell_rects:
            return []
        
        # Sort by position (top-to-bottom, left-to-right)
        cell_rects.sort(key=lambda r: (r[1], r[0]))
        
        # Group into rows by y-coordinate proximity
        rows = []
        current_row = [cell_rects[0]]
        
        for rect in cell_rects[1:]:
            if abs(rect[1] - current_row[0][1]) < 20:
                current_row.append(rect)
            else:
                current_row.sort(key=lambda r: r[0])
                rows.append(current_row)
                current_row = [rect]
        
        if current_row:
            current_row.sort(key=lambda r: r[0])
            rows.append(current_row)
        
        logger.info(f"Fallback detection: {len(rows)} rows")
        
        return rows
