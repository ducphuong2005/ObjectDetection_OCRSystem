
import os
import sys
import json
import logging
import copy
import numpy as np
import torch
import cv2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.utils.logger import setup_logger

from .config import DETECTION_CONFIG, CLASSES, get_project_root

setup_logger()
logger = logging.getLogger(__name__)


def register_datasets():
    """Đăng ký tập dữ liệu BOM (train/val) format COCO vào Detectron2."""
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data")

    for split in ["train", "val"]:
        dataset_name = f"bom_{split}"
        json_path = os.path.join(data_dir, "splits", f"{split}.json")
        image_root = os.path.join(data_dir, "raw", "BOM-Dataset")

        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)

        from detectron2.data.datasets import register_coco_instances
        register_coco_instances(dataset_name, {}, json_path, image_root)
        MetadataCatalog.get(dataset_name).set(thing_classes=CLASSES)

    logger.info("Đã đăng ký datasets: bom_train, bom_val")


class AugmentedTrainer(DefaultTrainer):
    """Trainer tuỳ chỉnh với augmentation cho ảnh bản vẽ kỹ thuật."""

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=(640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style="choice",
                ),
                T.RandomFlip(prob=0.3, horizontal=True),
                T.RandomRotation(angle=[-15, 15]),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup_cfg(resume=False):
    """Thiết lập config Detectron2 cho Faster R-CNN."""
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(DETECTION_CONFIG["model_zoo_config"])
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        DETECTION_CONFIG["model_zoo_config"]
    )

    cfg.DATASETS.TRAIN = (DETECTION_CONFIG["train_dataset"],)
    cfg.DATASETS.TEST = (DETECTION_CONFIG["val_dataset"],)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = DETECTION_CONFIG["num_classes"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_CONFIG["confidence_threshold"]

    cfg.SOLVER.IMS_PER_BATCH = DETECTION_CONFIG["batch_size"]
    cfg.SOLVER.BASE_LR = DETECTION_CONFIG["base_lr"]
    cfg.SOLVER.MAX_ITER = DETECTION_CONFIG["max_iter"]
    cfg.SOLVER.WARMUP_ITERS = DETECTION_CONFIG["warmup_iters"]
    cfg.SOLVER.WARMUP_FACTOR = DETECTION_CONFIG["warmup_factor"]
    cfg.SOLVER.STEPS = DETECTION_CONFIG["steps"]
    cfg.SOLVER.GAMMA = DETECTION_CONFIG["gamma"]
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = DETECTION_CONFIG["output_dir"]
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
        logger.warning("Không có CUDA, sử dụng CPU (sẽ chậm)")

    if resume:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    return cfg


def train(resume=False):
    """Huấn luyện model detection."""
    register_datasets()
    cfg = setup_cfg(resume=resume)

    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=resume)

    logger.info("Bắt đầu huấn luyện...")
    logger.info(f"  Model: {DETECTION_CONFIG['model_zoo_config']}")
    logger.info(f"  Classes: {CLASSES}")
    logger.info(f"  Max iterations: {DETECTION_CONFIG['max_iter']}")
    logger.info(f"  Learning rate: {DETECTION_CONFIG['base_lr']}")
    logger.info(f"  Output dir: {cfg.OUTPUT_DIR}")

    trainer.train()
    logger.info("Huấn luyện hoàn tất!")
    return cfg


def evaluate(cfg=None):
    """Đánh giá model trên tập validation."""
    if cfg is None:
        register_datasets()
        cfg = setup_cfg()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(
        DETECTION_CONFIG["val_dataset"],
        output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"),
    )
    val_loader = build_detection_test_loader(cfg, DETECTION_CONFIG["val_dataset"])
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    logger.info(f"Kết quả đánh giá:\n{results}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Huấn luyện BOM Object Detection")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint cuối")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá")
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        cfg = train(resume=args.resume)
        evaluate(cfg)
