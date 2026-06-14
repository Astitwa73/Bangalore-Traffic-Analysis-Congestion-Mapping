"""
src/data/preprocess.py
──────────────────────
Full preprocessing pipeline:
  1. Extract frames from VisDrone video sequences
  2. Parse VisDrone annotations → YOLO format
  3. Apply augmentations (albumentations)
  4. Generate train/val split
  5. Write dataset.yaml for YOLOv8
"""

import os
import cv2
import yaml
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import albumentations as A


# ── VisDrone label format ──────────────────────────────────────────────────────
# <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
# <score>,<object_category>,<truncation>,<occlusion>
VISDRONE_CATEGORIES = {
    0: "ignored",
    1: "pedestrian",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# Default mappings (will be overridden by config if present)
DEFAULT_VEHICLE_CLASS_MAP = {
    4: 0,   # car   → 0
    5: 1,   # van   → 1
    6: 2,   # truck → 2
    9: 3,   # bus   → 3
}

DEFAULT_YOLO_CLASS_NAMES = ["car", "van", "truck", "bus"]


# ── Augmentation pipeline ──────────────────────────────────────────────────────
def get_augmentation_pipeline() -> A.Compose:
    """Albumentations pipeline for aerial imagery."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.6),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CLAHE(clip_limit=4.0, p=0.3),   # contrast-limited adaptive histogram
            A.RandomShadow(p=0.2),             # simulate cloud shadows
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ── Frame extractor ────────────────────────────────────────────────────────────
class VisDronePreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = Path(self.cfg["dataset"]["raw_dir"])
        self.frames_dir = Path(self.cfg["dataset"]["frames_dir"])
        self.processed_dir = Path(self.cfg["dataset"]["processed_dir"])
        self.frame_skip = self.cfg["dataset"]["frame_skip"]
        self.img_size = self.cfg["dataset"]["img_size"]
        self.augment = self.cfg["training"]["augment"]
        self.aug_pipeline = get_augmentation_pipeline()

        # Load vehicle mappings from config or use defaults
        self.vehicle_class_map = self.cfg["dataset"].get("yolo_class_map", DEFAULT_VEHICLE_CLASS_MAP)
        # Ensure keys are integers if they came from YAML as strings
        self.vehicle_class_map = {int(k): v for k, v in self.vehicle_class_map.items()}
        
        # Load class names from config or derive from yolo_class_map
        if "yolo_class_names" in self.cfg["dataset"]:
            self.class_names = self.cfg["dataset"]["yolo_class_names"]
        else:
            # Fallback to defaults if they match the map, otherwise generic labels
            self.class_names = DEFAULT_YOLO_CLASS_NAMES if self.vehicle_class_map == DEFAULT_VEHICLE_CLASS_MAP else [f"class_{i}" for i in range(max(self.vehicle_class_map.values()) + 1)]

    # ── Step 1: Extract frames from sequences ─────────────────────────────────
    def extract_frames(self, split: str = "train") -> int:
        """
        Extract frames from VisDrone video sequences.
        Returns the number of extracted frames.
        """
        # Common patterns for VisDrone directories
        possible_roots = [
            self.raw_dir / f"VisDrone2019-VID-{split}",
            self.raw_dir / f"VisDrone2019-VID-{split}-val" if split == "val" else self.raw_dir,
            self.raw_dir / split,
            self.raw_dir
        ]
        
        seq_root = None
        ann_root = None
        
        for root in possible_roots:
            if (root / "sequences").exists() and (root / "annotations").exists():
                seq_root = root / "sequences"
                ann_root = root / "annotations"
                break
        
        if not seq_root:
            # Try recursive search if direct paths fail (useful for varied Kaggle uploads)
            seq_dirs = list(self.raw_dir.rglob("sequences"))
            if seq_dirs:
                seq_root = seq_dirs[0]
                ann_root = seq_root.parent / "annotations"

        if not seq_root or not seq_root.exists():
            raise FileNotFoundError(
                f"Could not find VisDrone 'sequences' directory in {self.raw_dir}\n"
                "Please check the dataset structure."
            )

        out_img_dir = self.frames_dir / split / "images"
        out_ann_dir = self.frames_dir / split / "annotations_raw"

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)

        sequences = sorted(seq_root.iterdir())
        total_frames = 0

        for seq_dir in tqdm(sequences, desc=f"Extracting [{split}]"):
            if not seq_dir.is_dir():
                continue
            ann_file = ann_root / f"{seq_dir.name}.txt"
            ann_data = self._load_visdrone_annotations(ann_file) if ann_file.exists() else {}

            frames = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png"))
            for idx, frame_path in enumerate(frames):
                if idx % self.frame_skip != 0:
                    continue
                frame_id = int(frame_path.stem)
                out_name = f"{seq_dir.name}_{frame_id:07d}"

                # Copy image
                dst_img = out_img_dir / f"{out_name}.jpg"
                if not dst_img.exists():
                    shutil.copy2(frame_path, dst_img)

                # Save raw annotation for this frame
                if frame_id in ann_data:
                    dst_ann = out_ann_dir / f"{out_name}.txt"
                    with open(dst_ann, "w") as f:
                        for row in ann_data[frame_id]:
                            f.write(",".join(map(str, row)) + "\n")

                total_frames += 1

        print(f"  [{split}] Extracted {total_frames} frames → {out_img_dir}")
        return total_frames

    # ── Step 2: Convert annotations → YOLO format ─────────────────────────────
    def convert_to_yolo(self, split: str = "train", augment: bool = False) -> int:
        """
        Read raw VisDrone annotations, filter vehicle classes,
        convert to YOLO normalized xywh format.
        """
        img_dir = self.frames_dir / split / "images"
        ann_dir = self.frames_dir / split / "annotations_raw"
        yolo_img_dir = self.processed_dir / split / "images"
        yolo_lbl_dir = self.processed_dir / split / "labels"

        yolo_img_dir.mkdir(parents=True, exist_ok=True)
        yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(img_dir.glob("*.jpg"))
        converted = 0

        for img_path in tqdm(image_files, desc=f"Converting [{split}]"):
            ann_path = ann_dir / f"{img_path.stem}.txt"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            yolo_labels = []
            if ann_path.exists():
                yolo_labels = self._raw_ann_to_yolo(ann_path, w, h)

            if not yolo_labels:   # skip empty frames during training
                continue

            # Resize image
            img_resized = cv2.resize(img, (self.img_size, self.img_size))

            # Write original
            out_name = img_path.stem
            cv2.imwrite(str(yolo_img_dir / f"{out_name}.jpg"), img_resized)
            self._write_yolo_labels(yolo_lbl_dir / f"{out_name}.txt", yolo_labels)
            converted += 1

            # Augmentation (train split only)
            if augment and split == "train":
                bboxes = [lbl[1:] for lbl in yolo_labels]
                class_ids = [lbl[0] for lbl in yolo_labels]
                aug_img, aug_labels = self._augment(img_resized, bboxes, class_ids)
                if aug_labels:
                    aug_name = f"{out_name}_aug"
                    cv2.imwrite(str(yolo_img_dir / f"{aug_name}.jpg"), aug_img)
                    self._write_yolo_labels(yolo_lbl_dir / f"{aug_name}.txt", aug_labels)

        print(f"  [{split}] {converted} labelled frames → {yolo_img_dir}")
        return converted

    # ── Step 3: Write dataset.yaml ─────────────────────────────────────────────
    def write_dataset_yaml(self) -> str:
        yaml_path = self.processed_dir / "dataset.yaml"
        content = {
            "path": str(self.processed_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False)
        print(f"\n  dataset.yaml → {yaml_path}")
        return str(yaml_path)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _load_visdrone_annotations(self, ann_path: Path) -> Dict[int, List]:
        """Parse VisDrone annotation file → {frame_id: [rows]}."""
        data: Dict[int, List] = {}
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                frame_id = int(parts[0])
                row = [int(x) for x in parts[:8]]
                data.setdefault(frame_id, []).append(row)
        return data

    def _raw_ann_to_yolo(self, ann_path: Path, img_w: int, img_h: int) -> List[List]:
        """Convert raw annotation rows to YOLO format, filtering vehicles."""
        yolo_labels = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                _, _, x, y, bw, bh, score, cat = (int(p) for p in parts[:8])
                if cat not in self.vehicle_class_map:
                    continue
                if score == 0:   # VisDrone marks ignored regions with score=0
                    continue
                # Clip to image bounds
                x = max(0, x); y = max(0, y)
                bw = min(bw, img_w - x); bh = min(bh, img_h - y)
                if bw <= 0 or bh <= 0:
                    continue
                # Convert to YOLO normalized cx, cy, w, h
                cx = (x + bw / 2) / img_w
                cy = (y + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h
                cls = self.vehicle_class_map[cat]
                yolo_labels.append([cls, cx, cy, nw, nh])
        return yolo_labels

    def _write_yolo_labels(self, path: Path, labels: List[List]) -> None:
        with open(path, "w") as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

    def _augment(
        self, img: np.ndarray, bboxes: List, class_ids: List
    ) -> Tuple[np.ndarray, List]:
        try:
            result = self.aug_pipeline(
                image=img, bboxes=bboxes, class_labels=class_ids
            )
            aug_labels = [
                [c, *list(b)]
                for c, b in zip(result["class_labels"], result["bboxes"])
            ]
            return result["image"], aug_labels
        except Exception:
            return img, []

    # ── Full pipeline ──────────────────────────────────────────────────────────
    def run(self, splits: Tuple[str, ...] = ("train", "val")) -> str:
        print("\n═══════════════════════════════════════════")
        print("  Bangalore Traffic — Preprocessing Pipeline")
        print("═══════════════════════════════════════════\n")
        for split in splits:
            print(f"── {split.upper()} ─────────────────────────────────")
            self.extract_frames(split)
            self.convert_to_yolo(split, augment=(split == "train"))
        return self.write_dataset_yaml()


# ── Roboflow shortcut (alternative to manual VisDrone setup) ──────────────────
def download_roboflow_dataset(api_key: str, workspace: str, project: str,
                               version: int, output_dir: str = "data/processed") -> str:
    """
    Download a pre-annotated aerial vehicle dataset from Roboflow Universe.
    Returns path to dataset.yaml.

    Recommended project: roboflow.com/universe/projects/aerial-vehicles-ojjme
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("pip install roboflow")

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8", location=output_dir)
    return os.path.join(output_dir, "data.yaml")


if __name__ == "__main__":
    preprocessor = VisDronePreprocessor("config.yaml")
    dataset_yaml = preprocessor.run(splits=("train", "val"))
    print(f"\n✅ Preprocessing done. dataset.yaml: {dataset_yaml}")
