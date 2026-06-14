"""
src/visualization/visualizer.py
────────────────────────────────
Composites all analysis layers into a single output frame and
writes annotated output videos.

Layout (configurable):
  ┌─────────────────────────────┐
  │   Main frame + bboxes       │
  │   + density/congestion      │
  ├──────────┬──────────────────┤
  │ Stats HUD│ Mini density map │
  └──────────┴──────────────────┘
"""

import cv2
import numpy as np
import yaml
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.analysis.congestion_mapper import CONGESTION_LEVELS


# ── Colours / fonts ────────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL  = 0.45
FONT_MEDIUM = 0.6
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
DARK_BG     = (20, 20, 20)
ACCENT      = (0, 200, 255)


class FrameCompositor:
    """
    Composites detector annotations, KDE heatmap, congestion grid,
    and a statistics HUD into the final output frame.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        vcfg = cfg["visualization"]
        dcfg = cfg["density"]
        ccfg = cfg["congestion"]

        self.show_bboxes     = vcfg["show_bboxes"]
        self.show_density    = vcfg["show_density"]
        self.show_congestion = vcfg["show_congestion"]
        self.show_stats      = vcfg["show_stats"]
        self.heatmap_alpha   = dcfg["heatmap_alpha"]
        self.thresholds      = ccfg["thresholds"]

        # Performance tracking attributes
        self.start_time  = time.time()
        self.frame_count = 0
        self.fps         = 0.0

    # ── Master compose ─────────────────────────────────────────────────────────
    def compose(
        self,
        frame: np.ndarray,
        annotated_frame: np.ndarray,    # frame with bboxes drawn
        kde_density: np.ndarray,        # float32 H×W density map
        kde_estimator,                  # KDEDensityEstimator instance
        grid_mapper,                    # CongestionMapper instance
        smoothed_grid: np.ndarray,      # float32 grid_rows×grid_cols
        stats: Dict,
        predicted_count: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build the final composite frame.
        """
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.start_time = time.time()
            self.frame_count = 0

        H, W = frame.shape[:2]
        result = frame.copy()

        if predicted_count is not None:
            stats["predicted_vehicles"] = predicted_count

        # Layer 1: bounding boxes
        if self.show_bboxes:
            result = annotated_frame.copy()

        # Layer 2: KDE density heatmap
        if self.show_density and kde_density is not None:
            heatmap = kde_estimator.to_heatmap(kde_density)
            result = cv2.addWeighted(result, 1 - self.heatmap_alpha,
                                     heatmap, self.heatmap_alpha, 0)

        # Layer 3: congestion grid overlay
        if self.show_congestion and smoothed_grid is not None:
            result = grid_mapper.draw_congestion_overlay(
                result, smoothed_grid, alpha=0.35, draw_grid_lines=False
            )
            result = grid_mapper.draw_legend(result)

        # Layer 4: stats HUD
        if self.show_stats and stats:
            result = self._draw_hud(result, stats)

        return result

    # ── HUD ────────────────────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        H, W = frame.shape[:2]

        # Semi-transparent top banner
        banner_h = 52
        banner = frame.copy()
        cv2.rectangle(banner, (0, 0), (W, banner_h), DARK_BG, -1)
        frame = cv2.addWeighted(frame, 0.35, banner, 0.65, 0)

        # Title
        cv2.putText(frame, "BANGALORE TRAFFIC ANALYSIS",
                    (10, 18), FONT, FONT_MEDIUM, ACCENT, 1, cv2.LINE_AA)
        
        # Live status (blinking effect)
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(frame, "• SCANNING", (W - 110, 18), FONT, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Metrics row
        n_veh = stats.get("n_vehicles", 0)
        p_veh = stats.get("predicted_vehicles", None)
        cidx  = stats.get("congestion_idx", 0.0)
        fno   = stats.get("frame", 0)

        # Congestion level label
        if cidx < 0.5:
            level_label, level_color = "FREE",     (0, 220, 80)
        elif cidx < 1.5:
            level_label, level_color = "MODERATE", (0, 220, 255)
        elif cidx < 2.5:
            level_label, level_color = "HEAVY",    (0, 140, 255)
        else:
            level_label, level_color = "SEVERE",   (0, 60, 255)

        metrics = [
            (f"Vehicles: {n_veh}", WHITE),
        ]
        
        if p_veh is not None:
            metrics.append((f"Forecast: {int(p_veh)}", (255, 200, 0)))

        metrics.extend([
            (f"Congestion: {level_label}", level_color),
            (f"Severe zones: {stats.get('pct_severe', 0.0):.1f}%", (0, 80, 255) if stats.get('pct_severe', 0.0) > 10 else WHITE),
            (f"FPS: {self.fps:.1f}", ACCENT),
            (f"Frame: {fno}", (160, 160, 160)),
        ])

        x = 10
        for text, color in metrics:
            cv2.putText(frame, text, (x, 42), FONT, FONT_SMALL, color, 1, cv2.LINE_AA)
            x += len(text) * 8 + 25   # improved spacing

        return frame

    # ── Mini thumbnail panel ───────────────────────────────────────────────────
    def draw_mini_panel(
        self,
        frame: np.ndarray,
        density_map: np.ndarray,
        panel_size: Tuple[int, int] = (160, 120),
    ) -> np.ndarray:
        """
        Draw a small density thumbnail in the bottom-right corner.
        """
        H, W = frame.shape[:2]
        pw, ph = panel_size

        if density_map is None:
            return frame

        heat = (density_map * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        thumb = cv2.resize(heat_color, (pw, ph))

        # Border
        cv2.rectangle(thumb, (0, 0), (pw - 1, ph - 1), ACCENT, 1)
        cv2.putText(thumb, "Density", (4, ph - 6),
                    FONT, 0.35, WHITE, 1)

        x1 = W - pw - 8
        y1 = H - ph - 8
        frame[y1:y1 + ph, x1:x1 + pw] = thumb
        return frame


# ── Video writer ───────────────────────────────────────────────────────────────
class VideoWriter:
    """Wraps cv2.VideoWriter with context manager support."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: str = "mp4v",
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            frame_size,
        )
        self.output_path = output_path
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        self.writer.release()
        print(f"  Video saved → {self.output_path} ({self.frames_written} frames)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# ── CSV stats logger ───────────────────────────────────────────────────────────
class StatsLogger:
    """Logs per-frame statistics to a CSV file."""

    FIELDS = [
        "frame", "timestamp_s", "n_vehicles",
        "n_car", "n_van", "n_truck", "n_bus",
        "pct_free", "pct_moderate", "pct_heavy", "pct_severe",
        "congestion_idx",
    ]

    def __init__(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self.output_path = output_path

    def log(self, frame_idx: int, fps: float, stats: Dict, class_counts: Dict) -> None:
        row = {
            "frame":          frame_idx,
            "timestamp_s":    round(frame_idx / max(fps, 1), 2),
            "n_vehicles":     stats.get("n_vehicles", 0),
            "n_car":          class_counts.get("car", 0),
            "n_van":          class_counts.get("van", 0),
            "n_truck":        class_counts.get("truck", 0),
            "n_bus":          class_counts.get("bus", 0),
            "pct_free":       round(stats.get("pct_free", 0), 2),
            "pct_moderate":   round(stats.get("pct_moderate", 0), 2),
            "pct_heavy":      round(stats.get("pct_heavy", 0), 2),
            "pct_severe":     round(stats.get("pct_severe", 0), 2),
            "congestion_idx": round(stats.get("congestion_idx", 0), 4),
        }
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()
        print(f"  Stats CSV → {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Summary image generator ───────────────────────────────────────────────────
def save_summary_heatmap(
    cumulative_heatmap: np.ndarray,
    output_path: str,
    title: str = "Bangalore Traffic Hotspots",
) -> None:
    """Save a labelled cumulative density heatmap as a PNG."""
    img = cumulative_heatmap.copy()
    H, W = img.shape[:2]

    # Title bar
    cv2.rectangle(img, (0, 0), (W, 40), DARK_BG, -1)
    cv2.putText(img, title, (10, 28), FONT, FONT_MEDIUM, ACCENT, 1, cv2.LINE_AA)

    # Colour scale bar (right side)
    bar_w, bar_h = 20, H - 60
    bar_x = W - bar_w - 10
    bar_y = 50
    for i in range(bar_h):
        val = int(255 * (1 - i / bar_h))
        color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
                                  cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.line(img, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), color, 1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  WHITE, 1)
    cv2.putText(img, "High", (bar_x - 28, bar_y + 10),
                FONT, 0.35, WHITE, 1)
    cv2.putText(img, "Low",  (bar_x - 22, bar_y + bar_h - 4),
                FONT, 0.35, WHITE, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"  Summary heatmap → {output_path}")
