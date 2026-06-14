"""
src/training/train.py
──────────────────────
Fine-tune YOLOv8 on the preprocessed vehicle dataset.
Includes:
  - YAML-driven config
  - Automatic GPU/CPU detection
  - Post-training metrics summary
  - Best model export to ONNX (optional)
"""

import yaml
import torch
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device(cfg_device) -> str:
    """Pick the best available device."""
    if cfg_device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return str(cfg_device)
    print("  No GPU found → using CPU (training will be slow).")
    return "cpu"


def train(config_path: str = "config.yaml", dataset_yaml: str = None) -> str:
    """
    Train YOLOv8 and return path to best weights.

    Parameters
    ----------
    config_path : str
        Path to project config.yaml.
    dataset_yaml : str | None
        Path to the dataset YAML produced by preprocess.py.
        If None, uses data/processed/dataset.yaml.

    Returns
    -------
    str
        Path to best.pt weights file.
    """
    cfg = load_config(config_path)
    tcfg = cfg["training"]

    dataset_yaml = dataset_yaml or str(
        Path(cfg["dataset"]["processed_dir"]) / "dataset.yaml"
    )
    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Run preprocess.py first."
        )

    device = get_device(tcfg["device"])
    run_name = f"bangalore_traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n═══════════════════════════════════════════")
    print("  YOLOv8 Training — Bangalore Traffic CV")
    print("═══════════════════════════════════════════")
    print(f"  Model   : {tcfg['model']}")
    print(f"  Epochs  : {tcfg['epochs']}")
    print(f"  Batch   : {tcfg['batch_size']}")
    print(f"  Device  : {device}")
    print(f"  Data    : {dataset_yaml}")
    print(f"  Run     : {run_name}\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = YOLO(tcfg["model"])

    # ── Train ──────────────────────────────────────────────────────────────────
    results = model.train(
        data=dataset_yaml,
        epochs=tcfg["epochs"],
        imgsz=tcfg["img_size"],
        batch=tcfg["batch_size"],
        lr0=tcfg["lr0"],
        lrf=tcfg["lrf"],
        momentum=tcfg["momentum"],
        weight_decay=tcfg["weight_decay"],
        patience=tcfg["patience"],
        device=device,
        workers=tcfg["workers"],
        augment=tcfg["augment"],
        project=tcfg["runs_dir"],
        name=run_name,
        exist_ok=True,
        verbose=True,
        # Aerial-specific tweaks
        degrees=45.0,        # random rotation (vehicles appear from any angle)
        scale=0.5,           # random scale
        mosaic=1.0,          # mosaic augmentation
        close_mosaic=10,     # disable last 10 epochs for stable training
        overlap_mask=False,
        plots=True,          # save training curves
    )

    best_weights = Path(tcfg["runs_dir"]) / run_name / "weights" / "best.pt"
    print_metrics(results, best_weights)

    return str(best_weights)


def print_metrics(results, best_weights: Path) -> None:
    """Print a clean summary of training results."""
    print("\n── Training Complete ──────────────────────────────────────")
    try:
        metrics = results.results_dict
        print(f"  mAP@0.5       : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95  : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision     : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall        : {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass
    print(f"  Best weights  : {best_weights}")
    print("───────────────────────────────────────────────────────────")


def export_onnx(weights_path: str, img_size: int = 640) -> str:
    """
    Export best.pt → best.onnx for deployment.
    ONNX is faster for CPU inference and cross-platform.
    """
    model = YOLO(weights_path)
    model.export(format="onnx", imgsz=img_size, simplify=True)
    onnx_path = weights_path.replace(".pt", ".onnx")
    print(f"  Exported ONNX model → {onnx_path}")
    return onnx_path


def validate(weights_path: str, dataset_yaml: str, config_path: str = "config.yaml") -> dict:
    """Run YOLOv8 validation on the val split."""
    cfg = load_config(config_path)
    icfg = cfg["inference"]
    device = get_device(cfg["training"]["device"])

    model = YOLO(weights_path)
    metrics = model.val(
        data=dataset_yaml,
        imgsz=cfg["dataset"]["img_size"],
        conf=icfg["conf_threshold"],
        iou=icfg["iou_threshold"],
        device=device,
        verbose=True,
    )
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None, help="Path to dataset.yaml")
    parser.add_argument("--export_onnx", action="store_true")
    args = parser.parse_args()

    best_weights = train(args.config, args.data)

    if args.export_onnx:
        export_onnx(best_weights)

    print(f"\n✅ Training done. Best weights: {best_weights}")
