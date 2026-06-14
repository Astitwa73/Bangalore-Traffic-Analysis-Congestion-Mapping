"""
src/inference/detector.py
──────────────────────────
YOLOv8-based vehicle detector for aerial/satellite frames.
Wraps ultralytics YOLO + supervision for clean detection output.

Usage:
    detector = VehicleDetector("runs/train/weights/best.pt")
    detections = detector.detect(frame)
"""

import cv2
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO
import supervision as sv


CLASS_NAMES = ["car", "van", "truck", "bus"]

# Draw colour per class (BGR)
CLASS_COLORS = {
    0: (0, 200, 255),    # car    — cyan
    1: (0, 255, 150),    # van    — green
    2: (0, 100, 255),    # truck  — orange
    3: (60,  60, 255),   # bus    — red
}


class VehicleDetector:
    """
    Detects vehicles in a single frame using a fine-tuned YOLOv8 model.

    Parameters
    ----------
    weights_path : str
        Path to best.pt (or best.onnx for ONNX runtime).
    conf : float
        Confidence threshold.
    iou : float
        NMS IoU threshold.
    imgsz : int
        Inference image size.
    device : str
        'cpu', '0', '0,1', etc.
    """

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        self.model   = YOLO(weights_path)
        self.conf    = conf
        self.iou     = iou
        self.imgsz   = imgsz
        self.device  = device
        self.classes = CLASS_NAMES

        # supervision annotators
        self.box_annotator = sv.BoxAnnotator(thickness=1)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.35,
            text_thickness=1,
            text_padding=2,
        )

    # ── Single frame detection ────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        sv.Detections  — xyxy, confidence, class_id arrays
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=300,
        )
        detections = sv.Detections.from_ultralytics(results[0])
        return detections

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        show_conf: bool = True,
    ) -> np.ndarray:
        """Draw detections on frame and return annotated copy."""
        annotated = frame.copy()

        labels = []
        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            conf   = float(detections.confidence[i])
            name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            label  = f"{name} {conf:.2f}" if show_conf else name
            labels.append(label)

        # Colour per class
        colors = sv.ColorPalette.from_hex([
            "#00C8FF",   # car
            "#00FF96",   # van
            "#FF6400",   # truck
            "#FF3C3C",   # bus
        ])

        annotated = self.box_annotator.annotate(
            scene=annotated, detections=detections
        )
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )
        return annotated

    # ── Convenience stats ─────────────────────────────────────────────────────
    def count_by_class(self, detections: sv.Detections) -> dict:
        counts = {name: 0 for name in CLASS_NAMES}
        if len(detections) == 0:
            return counts
        for cls_id in detections.class_id:
            name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "other"
            counts[name] = counts.get(name, 0) + 1
        return counts


class VideoProcessor:
    """
    Processes a full video file frame-by-frame.
    Yields (frame_idx, original_frame, detections) on each iteration.
    """

    def __init__(self, video_path: str, frame_skip: int = 1):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                yield frame_idx, frame
            frame_idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    @property
    def info(self) -> dict:
        return {
            "path":   self.video_path,
            "frames": self.total_frames,
            "fps":    self.fps,
            "size":   (self.width, self.height),
        }


def load_detector_from_config(config_path: str = "config.yaml") -> VehicleDetector:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    icfg = cfg["inference"]
    tcfg = cfg["training"]
    return VehicleDetector(
        weights_path=icfg["model_path"],
        conf=icfg["conf_threshold"],
        iou=icfg["iou_threshold"],
        imgsz=cfg["dataset"]["img_size"],
        device=str(tcfg["device"]),
    )
