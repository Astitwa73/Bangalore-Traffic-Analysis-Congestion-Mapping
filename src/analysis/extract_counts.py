"""
src/analysis/extract_counts.py
─────────────────────────────
Utility to run YOLO detection on video sequences and extract
per-frame vehicle counts into a CSV file for LSTM training.
"""

import argparse
import yaml
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.inference.detector import load_detector_from_config, VideoProcessor

def extract_counts(
    video_path: str,
    config_path: str = "config.yaml",
    output_csv: str = None
):
    """
    Process a video, detect vehicles, and save counts to CSV.
    """
    detector = load_detector_from_config(config_path)
    
    vid_path = Path(video_path)
    if output_csv is None:
        output_csv = f"data/counts_{vid_path.stem}.csv"
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    counts_data = []
    
    with VideoProcessor(video_path, frame_skip=1) as vp:
        print(f"Processing {vid_path.name}...")
        for frame_idx, frame in tqdm(vp, total=vp.total_frames):
            detections = detector.detect(frame)
            n_vehicles = len(detections)
            
            counts_data.append({
                "frame": frame_idx,
                "n_vehicles": n_vehicles
            })
            
    df = pd.DataFrame(counts_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Extracted counts for {len(df)} frames → {output_csv}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vehicle counts from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default=None, help="Path to output CSV")
    args = parser.parse_args()
    
    extract_counts(args.video, args.config, args.output)
