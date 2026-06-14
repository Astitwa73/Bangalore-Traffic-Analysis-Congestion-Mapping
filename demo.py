"""
demo.py
────────
One-click demonstration for Bangalore Traffic CV.
Processes a sample video and opens the results.
"""

import subprocess
import os
import sys
from pathlib import Path
import time

def run_demo():
    print("🚀 Initializing Bangalore Traffic CV Demo...")
    
    # Check for sample video
    sample_video = "data/sample.mp4"
    if not Path(sample_video).exists():
        # Try finding any mp4 in data/
        videos = list(Path("data").glob("*.mp4"))
        if videos:
            sample_video = str(videos[0])
        else:
            print(f"❌ Error: No video found in data/ folder.")
            return

    # Run the pipeline
    cmd = [
        sys.executable, "run_pipeline.py",
        "--mode", "infer",
        "--video", sample_video,
        "--preview"
    ]
    
    print(f"🎬 Processing: {sample_video}")
    print(f"💡 Press 'q' on the preview window to stop early.")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error during pipeline execution.")
        return

    # Results location
    video_stem = Path(sample_video).stem
    out_dir = Path("outputs")
    trend_map = out_dir / f"{video_stem}_trends.png"
    hotspots = out_dir / f"{video_stem}_hotspots.png"

    print("\n✅ Demo Complete!")
    print(f"📂 Results saved to: {out_dir.absolute()}")
    
    # Open outputs folder
    if os.name == 'nt':
        os.startfile(out_dir)
        # Open trend report
        if trend_map.exists():
            os.startfile(trend_map)
    else:
        subprocess.run(["xdg-open", str(out_dir)])
        if trend_map.exists():
            subprocess.run(["xdg-open", str(trend_map)])

if __name__ == "__main__":
    run_demo()
