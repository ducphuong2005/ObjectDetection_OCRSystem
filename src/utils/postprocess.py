
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class OCRPostProcessor:
    @staticmethod
    def fix_numeric_confusion(text: str) -> str:
        """Sửa nhầm ký tự trong ngữ cảnh số: O→0, l→1, S→5."""
        def replace_in_numeric(match):
            s = match.group()
            result = ""
            for ch in s:
                if ch in "OoO":
                    result += "0"
                elif ch in "lI|":
                    result += "1"
                elif ch in "SsS" and ch.isupper():
                    result += "5"
                else:
                    result += ch
            return result

        text = re.sub(r'\d+[OolI|]+\d*', replace_in_numeric, text)
        text = re.sub(r'[OolI|]+\d+', replace_in_numeric, text)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Chuẩn hoá khoảng trắng."""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\s*([,;:.])\s*', r'\1 ', text)
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)

    @staticmethod
    def fix_common_patterns(text: str) -> str:
        """Sửa pattern thường gặp trong bản vẽ: ø, ×, °."""
        text = re.sub(r'[øoO](\d)', r'ø\1', text)
        text = re.sub(r'(\d+)\s*[xX×]\s*(\d+)', r'\1×\2', text)
        text = re.sub(r'(\d+)\s*[°o]\s', r'\1° ', text)
        return text

    @staticmethod
    def clean_table_text(text: str) -> str:
        """Làm sạch text trích xuất từ ô bảng."""
        text = text.strip(' \t\n\r|_-')
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        return text

    def process(self, text: str, context: str = "general") -> str:
        """Pipeline hậu xử lý đầy đủ."""
        if not text:
            return ""
        if context == "table":
            text = self.clean_table_text(text)
            text = self.fix_numeric_confusion(text)
        elif context == "note":
            text = self.normalize_whitespace(text)
        text = self.fix_common_patterns(text)
        text = self.normalize_whitespace(text)
        return text

    def process_table(self, table_data: List[List[str]]) -> List[List[str]]:
        """Hậu xử lý tất cả ô trong bảng."""
        return [
            [self.process(cell, context="table") for cell in row]
            for row in table_data
        ]
