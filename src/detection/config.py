
import os

CLASSES = ["PartDrawing", "Note", "Table"]
NUM_CLASSES = len(CLASSES)

# Màu hiển thị (BGR cho OpenCV)
CLASS_COLORS = {
    "PartDrawing": (255, 255, 0),
    "Note": (128, 0, 128),
    "Table": (0, 0, 255),
}

DETECTION_CONFIG = {
    "model_zoo_config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "num_classes": NUM_CLASSES,

    "base_lr": 0.00025,
    "max_iter": 3000,
    "batch_size": 2,
    "warmup_iters": 200,
    "warmup_factor": 1.0 / 200,
    "steps": (2000, 2500),
    "gamma": 0.1,

    "train_dataset": "bom_train",
    "val_dataset": "bom_val",
    "images_per_batch": 2,

    "confidence_threshold": 0.5,
    "nms_threshold": 0.5,

    "output_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "detection"),
    "data_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
}


def get_project_root():
    """Trả về đường dẫn thư mục gốc của project."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
