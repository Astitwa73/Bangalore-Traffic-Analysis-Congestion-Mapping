"""
data/download_dataset.py
────────────────────────
Downloads VisDrone-VID (recommended) or DOTA v1.5 datasets.
VisDrone is the best free public dataset for aerial vehicle detection —
it was captured by drones over Chinese cities (similar urban density to Bangalore).
"""

import os
import zipfile
import shutil
import requests
import gdown
from pathlib import Path
from tqdm import tqdm


# ── VisDrone-VID Google Drive IDs ─────────────────────────────────────────────
# Source: https://github.com/VisDrone/VisDrone-Dataset
VISDRONE_URLS = {
    "VisDrone2019-VID-train": "1z_Y6T-tWRfa2HGv_1aQFMcAlnePFLGCJ",   # ~7 GB
    "VisDrone2019-VID-val":   "1NJE_xSQmFVhOB8TMG6OMuWdHOxaVHZRM",   # ~1.5 GB
    "VisDrone2019-VID-test":  "1BQ6QFiSmkDOoaUt5tkbgZxW2MjbS_OeY",   # ~2 GB
}

# ── DOTA v1.5 (satellite, harder but more satellite-like) ──────────────────────
DOTA_INFO = {
    "url": "https://captain-whu.github.io/DOTA/dataset.html",
    "note": "DOTA requires manual registration at the above URL.",
}


def download_file_gdrive(file_id: str, dest_path: str) -> None:
    """Download a file from Google Drive using gdown."""
    print(f"  Downloading to {dest_path} ...")
    gdown.download(id=file_id, output=dest_path, quiet=False)


def download_file_http(url: str, dest_path: str) -> None:
    """Stream-download a file over HTTP with a progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=Path(dest_path).name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: str, dest_dir: str) -> None:
    print(f"  Extracting {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)
    print("  Done. Zip removed.")


def download_visdrone(raw_dir: str = "data/raw", splits=("train", "val")) -> None:
    """
    Download selected VisDrone-VID splits.

    Parameters
    ----------
    raw_dir : str
        Root directory where data will be saved.
    splits : tuple
        Which splits to download: 'train', 'val', 'test'.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        key = f"VisDrone2019-VID-{split}"
        if key not in VISDRONE_URLS:
            print(f"[SKIP] Unknown split: {split}")
            continue

        dest_dir = raw_dir / key
        if dest_dir.exists():
            print(f"[SKIP] {key} already downloaded at {dest_dir}")
            continue

        zip_path = str(raw_dir / f"{key}.zip")
        print(f"\n[DOWNLOAD] {key}")
        download_file_gdrive(VISDRONE_URLS[key], zip_path)
        extract_zip(zip_path, str(raw_dir))

    print("\n✅ VisDrone download complete.")
    print_dataset_info(raw_dir)


def print_dataset_info(raw_dir: Path) -> None:
    print("\n── Dataset Summary ──────────────────────────────────────")
    for split_dir in sorted(raw_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        seq_dirs = list((split_dir / "sequences").glob("*")) if (split_dir / "sequences").exists() else []
        ann_files = list((split_dir / "annotations").glob("*.txt")) if (split_dir / "annotations").exists() else []
        print(f"  {split_dir.name}: {len(seq_dirs)} sequences, {len(ann_files)} annotation files")
    print("─────────────────────────────────────────────────────────")


def print_alternative_datasets() -> None:
    """Print a guide to alternative datasets."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              Alternative / Supplementary Datasets                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. VisDrone-DET (images only, faster to work with)              ║
║     https://github.com/VisDrone/VisDrone-Dataset                 ║
║                                                                  ║
║  2. DOTA v1.5 / v2.0 (true satellite imagery)                    ║
║     https://captain-whu.github.io/DOTA/dataset.html              ║
║     ⚠️  Requires registration & manual download                   ║
║                                                                  ║
║  3. UAV123 (UAV tracking benchmark)                              ║
║     https://cemse.kaust.edu.sa/ivul/uav123                       ║
║                                                                  ║
║  4. Roboflow Universe — "aerial vehicles" (ready-to-use YOLO)    ║
║     https://universe.roboflow.com                                ║
║     ✅ Easiest for beginners — pre-labelled, YOLO format         ║
║                                                                  ║
║  5. Google Earth Engine (Bangalore-specific imagery)             ║
║     https://earthengine.google.com                               ║
║     Needs GEE account; good for spatial reference layers         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download aerial traffic datasets")
    parser.add_argument("--dataset", default="visdrone", choices=["visdrone", "info"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--raw_dir", default="data/raw")
    args = parser.parse_args()

    print_alternative_datasets()

    if args.dataset == "visdrone":
        download_visdrone(raw_dir=args.raw_dir, splits=args.splits)
    else:
        print("Use --dataset visdrone to start downloading.")
