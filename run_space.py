"""
Entry point for HuggingFace Spaces (Docker SDK).
Detectron2 đã được cài sẵn trong Dockerfile, không cần cài runtime.
"""
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import create_demo

# HuggingFace Spaces requires the `demo` variable in the global scope
# Pipeline sẽ được Lazy-load khi user bấm nút "Phát hiện" lần đầu
demo = create_demo()

# Khi chạy trực tiếp (Docker CMD), launch server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
