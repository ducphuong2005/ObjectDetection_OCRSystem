#  BOM Object Detection & OCR System

Hệ thống Computer Vision end-to-end để **phát hiện và trích xuất nội dung** từ bản vẽ kỹ thuật BOM (Bill of Materials).

## Pipeline Tổng quan

```
Ảnh bản vẽ kỹ thuật
   ↓
[1] Object Detection (Detectron2 Faster R-CNN)
   ↓
[2] Cắt vùng (Crop) từng đối tượng
   ↓
[3] OCR — Note: trích xuất text, Table: tái tạo cấu trúc bảng
   ↓
[4] Hậu xử lý OCR (sửa nhầm ký tự, chuẩn hoá)
   ↓
[5] Xuất JSON có cấu trúc
   ↓
[6] Web demo trực quan (Gradio)
```

## Cấu trúc thư mục

```
ObjectDetection_OCRSystem/
├── app/
│   └── app.py                 # Web demo Gradio
├── src/
│   ├── detection/
│   │   ├── config.py          # Cấu hình model và hyperparameters
│   │   ├── augmentation.py    # Augmentation cho training
│   │   ├── train.py           # Script huấn luyện Detectron2
│   │   └── inference.py       # Module inference
│   ├── ocr/
│   │   ├── note_ocr.py        # OCR vùng Note (PaddleOCR + Tesseract)
│   │   ├── table_ocr.py       # OCR bảng — position-based reconstruction
│   │   └── ensemble.py        # Kết hợp PaddleOCR + Tesseract
│   ├── utils/
│   │   ├── cropper.py         # Cắt vùng đối tượng phát hiện
│   │   ├── postprocess.py     # Hậu xử lý kết quả OCR
│   │   └── visualizer.py      # Vẽ bounding box và nhãn
│   └── pipeline/
│       └── pipeline.py        # Pipeline end-to-end
├── scripts/
│   ├── annotate.py            # Công cụ gán nhãn tương tác (OpenCV GUI)
│   ├── convert_to_coco.py     # Chuyển annotation sang COCO format
│   ├── evaluate.py            # Đánh giá model
│   └── test_ocr.py            # Test OCR trên crop
├── data/
│   ├── raw/BOM-Dataset/       # Ảnh gốc bản vẽ kỹ thuật
│   ├── annotations/           # Annotation từng ảnh (JSON)
│   └── splits/                # train.json, val.json (COCO format)
├── models/detection/          # Model weights (.pth)
├── outputs/                   # Kết quả xuất ra
│   ├── json/                  # JSON trích xuất
│   ├── crops/                 # Ảnh crop từng vùng
│   └── visualizations/        # Ảnh trực quan hoá
├── requirements.txt
├── .gitignore
└── README.md
```

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Tesseract OCR (`brew install tesseract` trên macOS)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cài Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Hướng dẫn sử dụng

### 1. Gán nhãn dữ liệu

```bash
python scripts/annotate.py --data-dir data/raw/BOM-Dataset --output-dir data/annotations
```

**Điều khiển:**
| Phím | Chức năng |
|------|-----------|
| Click trái + kéo | Vẽ bounding box |
| `1` / `2` / `3` | Chọn class: PartDrawing / Note / Table |
| `u` | Hoàn tác annotation cuối |
| `n` / `p` | Ảnh tiếp / trước |
| `s` | Lưu |
| `q` | Thoát |

### 2. Chuyển sang COCO format

```bash
python scripts/convert_to_coco.py
```

### 3. Huấn luyện model

```bash
python -m src.detection.train
```

Hoặc chỉ đánh giá:

```bash
python -m src.detection.train --eval-only
```

### 4. Đánh giá model

```bash
python scripts/evaluate.py --model models/detection/model_final.pth
```

### 5. Chạy Web Demo

```bash
python app/app.py
```

Mở trình duyệt tại `http://localhost:7860`

## 3 lớp đối tượng

| Class | Mô tả | Xử lý OCR |
|-------|--------|------------|
| 🔵 **PartDrawing** | Bản vẽ chi tiết | Không OCR |
| 🟣 **Note** | Ghi chú kỹ thuật | Trích xuất text đầy đủ |
| 🔴 **Table** | Bảng BOM | Tái tạo cấu trúc hàng/cột |

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Object Detection | **Detectron2** — Faster R-CNN R50-FPN |
| OCR chính | **PaddleOCR v5** |
| OCR fallback | **Tesseract** |
| Augmentation | **Albumentations** |
| Web Demo | **Gradio** |
| Xử lý ảnh | **OpenCV** |

## Kiến trúc OCR

### Table OCR — Position-based Reconstruction
Thay vì chia nhỏ từng ô (quá bé để OCR), hệ thống:
1. Chạy PaddleOCR trên **toàn bộ ảnh bảng**
2. **Nhóm text theo hàng** dựa trên vị trí Y (adaptive threshold)
3. **Align cột** bằng clustering cạnh trái (x1) với threshold tự động

### Note OCR — Dual-pass Strategy
1. Chạy PaddleOCR trên ảnh **đã tiền xử lý** (phóng to + tăng nét)
2. Chạy PaddleOCR trên ảnh **gốc**
3. Chọn kết quả có **confidence cao hơn**
4. Fallback sang **Tesseract** nếu cần

## Đầu ra JSON

```json
{
  "image": "drawing_01.jpg",
  "num_objects": 3,
  "objects": [
    {
      "id": 1,
      "class": "Table",
      "confidence": 0.99,
      "bbox": {"x1": 446.9, "y1": 15.1, "x2": 1030.0, "y2": 483.2},
      "ocr_content": [
        ["ITEM", "QTY.", "DESCRIPTION", "GRADE", "PART NUMBER", "MASS"],
        ["1", "1", "BUCKSTAY", "", "H368377-42020-...", "4592.3"]
      ]
    },
    {
      "id": 2,
      "class": "Note",
      "confidence": 0.98,
      "ocr_content": "THIS DRAWING SHALL BE READ WITH SPECIFICATION..."
    }
  ]
}
```