
import os
import sys
import json
import cv2
import numpy as np
import logging
import argparse
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src.pipeline.pipeline import BOMPipeline
from src.utils.visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None


def initialize_pipeline(model_path: Optional[str] = None):
    """Khởi tạo pipeline."""
    global pipeline
    pipeline = BOMPipeline(model_path=model_path, confidence_threshold=0.5, output_dir="outputs")


def process_image(input_image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, str, str, list]:
    """Xử lý ảnh upload qua pipeline, trả ảnh annotated + JSON + HTML OCR + gallery."""
    global pipeline

    if pipeline is None:
        initialize_pipeline()
    if input_image is None:
        return None, "{}", "Chưa upload ảnh", []

    pipeline.detector.confidence_threshold = confidence_threshold
    bgr_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    temp_path = "/tmp/bom_input_temp.jpg"
    cv2.imwrite(temp_path, bgr_image)

    try:
        result = pipeline.process(temp_path, save_outputs=True)
    except Exception as e:
        logger.error(f"Lỗi pipeline: {e}")
        return input_image, json.dumps({"error": str(e)}), f"Lỗi: {e}", []

    visualizer = Visualizer()
    vis_bgr = visualizer.draw_detections(bgr_image, result["objects"])
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    json_output = json.dumps(pipeline._make_serializable(result), ensure_ascii=False, indent=2)
    ocr_html = build_ocr_html(result["objects"])

    crops_gallery = []
    for obj in result["objects"]:
        crop_path = obj.get("crop_path")
        if crop_path and os.path.exists(crop_path):
            crop_img = cv2.imread(crop_path)
            if crop_img is not None:
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crops_gallery.append((crop_rgb, f"{obj['class']} #{obj['id']} ({obj['confidence']:.2f})"))

    return vis_rgb, json_output, ocr_html, crops_gallery


def build_ocr_html(objects: list) -> str:
    """Tạo HTML hiển thị kết quả OCR — bảng dạng <table>, Note dạng text block."""
    if not objects:
        return '<p style="color:#888;">Không phát hiện đối tượng nào</p>'

    parts = []
    for obj in objects:
        cls = obj["class"]
        conf = obj["confidence"]
        content = obj.get("ocr_content")
        obj_id = obj["id"]

        color = {"Table": "#ef4444", "Note": "#a855f7", "PartDrawing": "#06b6d4"}.get(cls, "#888")
        parts.append(
            f'<div style="margin:12px 0 6px; padding:6px 12px; '
            f'background:{color}22; border-left:4px solid {color}; '
            f'border-radius:4px; font-weight:600; color:{color};">'
            f'{cls} #{obj_id} &nbsp;'
            f'<span style="font-weight:400; font-size:0.85em; opacity:0.7;">'
            f'conf: {conf:.2f}</span></div>'
        )

        if cls == "Table" and isinstance(content, list) and content:
            parts.append(_build_html_table(content))
        elif cls == "Note" and content:
            escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            formatted = "<br>".join(escaped.split("\n"))
            parts.append(
                f'<div style="padding:8px 12px; background:#1e1e2e; '
                f'border-radius:6px; font-family:monospace; font-size:0.9em; '
                f'line-height:1.6; color:#cdd6f4; white-space:pre-wrap;">'
                f'{formatted}</div>'
            )
        elif cls == "PartDrawing":
            parts.append('<div style="padding:6px 12px; color:#666; font-style:italic;">'
                         '[Bản vẽ chi tiết — không cần OCR]</div>')

    return '\n'.join(parts)


def _build_html_table(table_data: list) -> str:
    """Tạo HTML table có style từ mảng 2D."""
    if not table_data:
        return ''

    html = [
        '<div style="overflow-x:auto; margin:4px 0;">',
        '<table style="border-collapse:collapse; width:100%; font-size:0.85em; font-family:monospace;">',
    ]

    for row_idx, row in enumerate(table_data):
        html.append('<tr>')
        for cell in row:
            cell_text = str(cell).strip() if cell else ''
            cell_text = cell_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            if row_idx == 0:
                html.append(
                    f'<th style="border:1px solid #444; padding:6px 8px; '
                    f'background:#2d2d3f; color:#89b4fa; text-align:left; '
                    f'font-weight:600; white-space:nowrap;">{cell_text}</th>'
                )
            else:
                bg = '#1e1e2e' if row_idx % 2 == 0 else '#181825'
                html.append(
                    f'<td style="border:1px solid #333; padding:4px 8px; '
                    f'background:{bg}; color:#cdd6f4; white-space:nowrap;">'
                    f'{cell_text}</td>'
                )
        html.append('</tr>')

    html.append('</table></div>')
    return '\n'.join(html)


def create_demo() -> gr.Blocks:
    """Tạo giao diện Gradio."""
    with gr.Blocks(title="BOM Detection & OCR System") as demo:
        gr.Markdown("""
            # Hệ thống Phát hiện & OCR Bản vẽ BOM

            Upload bản vẽ kỹ thuật để phát hiện các vùng **PartDrawing**, **Note**, **Table**
            và trích xuất nội dung text/bảng tự động.

            ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload bản vẽ", type="numpy", height=400)
                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.99, value=0.5, step=0.05,
                    label="Ngưỡng confidence",
                )
                run_btn = gr.Button("Phát hiện & Trích xuất", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Kết quả phát hiện", height=400)

        with gr.Row():
            with gr.Column(scale=1):
                json_output = gr.Code(label="JSON Output", language="json", lines=20)
            with gr.Column(scale=1):
                ocr_output = gr.HTML(
                    label="Kết quả OCR",
                    value='<p style="color:#888;">Upload ảnh và nhấn Phát hiện</p>',
                )

        with gr.Row():
            crops_gallery = gr.Gallery(label="Vùng đã cắt", columns=4, height=250)

        example_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset", "BOM-Dataset")
        if os.path.exists(example_dir):
            example_images = [
                os.path.join(example_dir, f)
                for f in sorted(os.listdir(example_dir))[:5]
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if example_images:
                gr.Examples(
                    examples=[[img, 0.5] for img in example_images],
                    inputs=[input_image, confidence_slider],
                    label="Ảnh mẫu",
                )

        run_btn.click(
            fn=process_image,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, json_output, ocr_output, crops_gallery],
        )

        gr.Markdown("""
            ---
            **Pipeline:** Detectron2 Faster R-CNN → Cắt vùng → PaddleOCR + Tesseract → Tái tạo cấu trúc bảng → JSON

            **Classes:** 🔵 PartDrawing | 🟣 Note | 🔴 Table
        """)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Demo BOM Detection")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn model weights")
    parser.add_argument("--share", action="store_true", help="Tạo link công khai")
    args = parser.parse_args()

    initialize_pipeline(args.model)
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        css=".main-title { text-align: center; margin-bottom: 20px; }",
    )
