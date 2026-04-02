
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.note_ocr import NoteOCR
from src.ocr.table_ocr import TableOCR


def test_note():
    crop_path = "outputs/crops/Note/bom_input_temp_note_1.png"
    if not os.path.exists(crop_path):
        print(f"Không tìm thấy: {crop_path}")
        return

    image = cv2.imread(crop_path)
    print(f"Note image: {image.shape}")
    ocr = NoteOCR(lang="en")
    text = ocr.extract(image, use_ensemble=False)
    print(f"\n=== Kết quả Note OCR ===\n{text}\n=== {len(text)} ký tự ===\n")


def test_table():
    for f in ["bom_input_temp_table_1.png", "bom_input_temp_table_2.png"]:
        crop_path = f"outputs/crops/Table/{f}"
        if not os.path.exists(crop_path):
            continue

        image = cv2.imread(crop_path)
        print(f"\nTable image: {image.shape} ({f})")
        ocr = TableOCR(lang="en")
        result = ocr.extract(image)
        print(f"=== Table OCR: {result['num_rows']} hàng × {result['num_cols']} cột ===")
        for i, row in enumerate(result['table_data']):
            print(f"  Hàng {i}: {row}")
        print("===\n")


if __name__ == "__main__":
    print("Test Note OCR...")
    test_note()
    print("Test Table OCR...")
    test_table()
