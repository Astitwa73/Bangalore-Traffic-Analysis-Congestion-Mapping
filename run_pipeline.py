"""
run_pipeline.py
────────────────
End-to-end Bangalore Traffic CV pipeline.

Modes:
  preprocess   — extract frames & convert to YOLO format
  train        — fine-tune YOLOv8
  infer        — run detection + density + congestion on a video
  all          — run all three steps sequentially

Usage examples:
  python run_pipeline.py --mode preprocess
  python run_pipeline.py --mode train
  python run_pipeline.py --mode infer --video path/to/video.mp4
  python run_pipeline.py --mode all   --video path/to/video.mp4
"""

import sys
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocess          import VisDronePreprocessor
from src.training.train           import train as train_model
from src.inference.detector       import VehicleDetector, VideoProcessor, load_detector_from_config
from src.analysis.density_estimator import (
    KDEDensityEstimator, GridDensityEstimator,
    get_centers_and_weights, load_estimators_from_config,
)
from src.analysis.congestion_mapper import (
    CongestionMapper, load_congestion_mapper_from_config,
)
from src.analysis.predict_lstm import TrafficPredictor
from src.visualization.visualizer import (
    FrameCompositor, VideoWriter, StatsLogger, save_summary_heatmap,
)
from src.analysis.report_generator import generate_trend_report
from collections import deque


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocess(config_path: str) -> str:
    print("\n╔══════════════════════════════════════╗")
    print("║  STEP 1 — PREPROCESSING               ║")
    print("╚══════════════════════════════════════╝")
    preprocessor = VisDronePreprocessor(config_path)
    dataset_yaml = preprocessor.run(splits=("train", "val"))
    print(f"\n✅  Preprocessing complete → {dataset_yaml}")
    return dataset_yaml


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_train(config_path: str, dataset_yaml: str = None) -> str:
    print("\n╔══════════════════════════════════════╗")
    print("║  STEP 2 — TRAINING                    ║")
    print("╚══════════════════════════════════════╝")
    best_weights = train_model(config_path, dataset_yaml)
    print(f"\n✅  Training complete → {best_weights}")
    return best_weights


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — INFERENCE + ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_infer(
    config_path: str,
    video_path: str,
    weights_path: str = None,
    lstm_weights: str = None,
    output_dir: str = None,
    show_preview: bool = False,
) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║  STEP 3 — INFERENCE & ANALYSIS        ║")
    print("╚══════════════════════════════════════╝")

    cfg = load_config(config_path)
    out_dir = Path(output_dir or cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Override weights path if provided ─────────────────────────────────────
    if weights_path:
        cfg["inference"]["model_path"] = weights_path
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

    # ── Initialise modules ─────────────────────────────────────────────────────
    detector    = load_detector_from_config(config_path)
    kde, grid   = load_estimators_from_config(config_path)
    mapper      = load_congestion_mapper_from_config(config_path)
    compositor  = FrameCompositor(config_path)

    # ── LSTM Predictor ─────────────────────────────────────────────────────────
    predictor = None
    count_buffer = None
    if lstm_weights and Path(lstm_weights).exists():
        print(f"  Loading LSTM Forecast Model: {lstm_weights}")
        predictor = TrafficPredictor(lstm_weights)
        count_buffer = deque(maxlen=predictor.seq_len)
    elif lstm_weights:
        print(f"  ⚠️ LSTM weights not found at {lstm_weights}. Skipping forecast.")

    # ── Open video ─────────────────────────────────────────────────────────────
    vid_name = Path(video_path).stem
    with VideoProcessor(video_path, frame_skip=1) as vp:
        print(f"\n  Video : {video_path}")
        print(f"  Size  : {vp.width}×{vp.height}  |  FPS: {vp.fps:.1f}  |  Frames: {vp.total_frames}")

        out_video_path = str(out_dir / f"{vid_name}_output.mp4")
        out_csv_path   = str(out_dir / f"{vid_name}_stats.csv")
        out_hmap_path  = str(out_dir / f"{vid_name}_hotspots.png")
        out_trend_path = str(out_dir / f"{vid_name}_trends.png")

        with (
            VideoWriter(out_video_path, vp.fps, (vp.width, vp.height)) as writer,
            StatsLogger(out_csv_path) as logger,
        ):
            for frame_idx, frame in tqdm(
                vp, total=vp.total_frames, desc="Processing"
            ):
                H, W = frame.shape[:2]

                # ── Detection ─────────────────────────────────────────────────
                detections    = detector.detect(frame)
                annotated     = detector.annotate(frame, detections)
                class_counts  = detector.count_by_class(detections)
                n_vehicles    = len(detections)

                # ── Forecasting ───────────────────────────────────────────────
                predicted_count = None
                if predictor:
                    count_buffer.append(float(n_vehicles))
                    if len(count_buffer) == predictor.seq_len:
                        preds = predictor.predict(list(count_buffer))
                        predicted_count = preds[0] # next frame forecast

                # ── Density estimation ─────────────────────────────────────────
                centers, weights = get_centers_and_weights(detections)
                kde_density      = kde.estimate((H, W), centers, weights)
                grid_counts, _   = grid.estimate((H, W), centers, weights)

                # ── Congestion mapping ─────────────────────────────────────────
                smoothed = mapper.update(grid_counts, n_vehicles)
                stats    = mapper.get_frame_stats(smoothed, n_vehicles)

                # ── Compose final frame ────────────────────────────────────────
                output_frame = compositor.compose(
                    frame, annotated,
                    kde_density, kde,
                    mapper, smoothed,
                    stats,
                    predicted_count=predicted_count
                )
                output_frame = compositor.draw_mini_panel(output_frame, kde_density)

                writer.write(output_frame)
                logger.log(frame_idx, vp.fps, stats, class_counts)

                # ── Optional live preview ──────────────────────────────────────
                if show_preview:
                    cv2.imshow("Bangalore Traffic CV", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if show_preview:
            cv2.destroyAllWindows()

    # ── Save cumulative hotspot map ────────────────────────────────────────────
    cumulative_hmap = mapper.draw_heatmap_cumulative(
        (vp.height, vp.width)
    )
    save_summary_heatmap(cumulative_hmap, out_hmap_path)

    # ── Generate trend report ──────────────────────────────────────────────────
    print(f"  Generating trend report...")
    generate_trend_report(out_csv_path, out_trend_path)
    print(f"  Trend analysis saved → {out_trend_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    summary = mapper.get_summary()
    print_summary(summary, out_dir)


def print_summary(summary: dict, out_dir: Path) -> None:
    print("\n── Analysis Summary ────────────────────────────────────────")
    print(f"  Frames processed  : {summary.get('total_frames', 0)}")
    print(f"  Avg vehicles/frame: {summary.get('avg_vehicles', 0):.1f}")
    print(f"  Peak vehicles     : {summary.get('peak_vehicles', 0)}  "
          f"(frame {summary.get('peak_frame', 0)})")
    print(f"  Hotspot cells     : {summary.get('hotspot_cells', 0)}")
    print(f"\n  Output files in   : {out_dir}/")
    print("───────────────────────────────────────────────────────────")
    print("\n✅  Inference complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_all(config_path: str, video_path: str, show_preview: bool = False) -> None:
    dataset_yaml  = run_preprocess(config_path)
    best_weights  = run_train(config_path, dataset_yaml)
    run_infer(config_path, video_path, best_weights, show_preview=show_preview)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Bangalore Traffic CV Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode preprocess
  python run_pipeline.py --mode train
  python run_pipeline.py --mode infer --video data/sample.mp4
  python run_pipeline.py --mode infer --video data/sample.mp4 --weights runs/train/weights/best.pt --preview
  python run_pipeline.py --mode all   --video data/sample.mp4
        """,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["preprocess", "train", "infer", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config",  default="config.yaml",  help="Config file path")
    parser.add_argument("--video",   default=None,           help="Input video (infer/all mode)")
    parser.add_argument("--weights", default=None,           help="Override model weights path")
    parser.add_argument("--lstm",    default=None,           help="Path to trained LSTM weights (lstm_congestion.pt)")
    parser.add_argument("--output",  default=None,           help="Override output directory")
    parser.add_argument("--preview", action="store_true",    help="Show live preview window")
    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocess(args.config)

    elif args.mode == "train":
        run_train(args.config)

    elif args.mode == "infer":
        if not args.video:
            parser.error("--video is required for infer mode")
        run_infer(args.config, args.video, args.weights, args.lstm, args.output, args.preview)

    elif args.mode == "all":
        if not args.video:
            parser.error("--video is required for all mode")
        run_all(args.config, args.video, args.preview)


if __name__ == "__main__":
    main()


