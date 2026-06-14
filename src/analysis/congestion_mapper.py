"""
src/analysis/congestion_mapper.py
──────────────────────────────────
Aggregates per-frame density estimates into:
  1. Temporal rolling average — smooths out single-frame noise
  2. Congestion level map — categorical: free / moderate / heavy / severe
  3. Per-zone statistics — average vehicle count, peak hour detection
  4. Summary statistics for dashboard
"""

from collections import deque
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import yaml


# ── Congestion Levels ──────────────────────────────────────────────────────────
CONGESTION_LEVELS = {
    "free":     {"color_bgr": (0, 200,   0), "label": "Free"},
    "moderate": {"color_bgr": (0, 220, 255), "label": "Moderate"},
    "heavy":    {"color_bgr": (0, 120, 255), "label": "Heavy"},
    "severe":   {"color_bgr": (0,   0, 255), "label": "Severe"},
}


class CongestionMapper:
    """
    Maintains a rolling buffer of density maps and computes congestion metrics.

    Parameters
    ----------
    grid_rows, grid_cols : int
        Size of the analysis grid.
    thresholds : dict
        Counts per cell defining {low, medium, high} boundaries.
    buffer_size : int
        How many past frames to average over (temporal smoothing).
    smoothing_sigma : float
        Gaussian sigma for spatial smoothing of the congestion map.
    """

    def __init__(
        self,
        grid_rows: int = 20,
        grid_cols: int = 20,
        thresholds: Optional[Dict] = None,
        buffer_size: int = 15,
        smoothing_sigma: float = 1.5,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.thresholds = thresholds or {"low": 2, "medium": 5, "high": 10}
        self.buffer_size = buffer_size
        self.sigma = smoothing_sigma

        # Rolling buffer of raw grid counts
        self._buffer: deque = deque(maxlen=buffer_size)

        # Cumulative stats
        self.frame_count = 0
        self.total_vehicles: List[int] = []           # vehicles per frame
        self.zone_totals = np.zeros((grid_rows, grid_cols), dtype=np.float64)

    # ── Core update ───────────────────────────────────────────────────────────
    def update(self, grid_counts: np.ndarray, n_vehicles: int) -> np.ndarray:
        """
        Accept a new grid count map and return the smoothed congestion map.

        Parameters
        ----------
        grid_counts : int array (grid_rows, grid_cols)
        n_vehicles  : total vehicles detected this frame

        Returns
        -------
        smoothed : float32 (grid_rows, grid_cols) — temporal average
        """
        self._buffer.append(grid_counts.astype(np.float32))
        self.frame_count += 1
        self.total_vehicles.append(n_vehicles)
        self.zone_totals += grid_counts.astype(np.float64)

        smoothed = np.mean(np.stack(self._buffer), axis=0)
        return smoothed.astype(np.float32)

    # ── Congestion level map ──────────────────────────────────────────────────
    def get_level_map(self, smoothed: np.ndarray) -> np.ndarray:
        """
        Classify each cell into a congestion level.

        Returns
        -------
        level_map : uint8 (grid_rows, grid_cols)
            0 = free, 1 = moderate, 2 = heavy, 3 = severe
        """
        low    = self.thresholds["low"]
        medium = self.thresholds["medium"]
        high   = self.thresholds["high"]

        level_map = np.zeros_like(smoothed, dtype=np.uint8)
        level_map[smoothed >= low]    = 1
        level_map[smoothed >= medium] = 2
        level_map[smoothed >= high]   = 3
        return level_map

    # ── Visualisation ─────────────────────────────────────────────────────────
    def draw_congestion_overlay(
        self,
        frame: np.ndarray,
        smoothed: np.ndarray,
        alpha: float = 0.45,
        draw_grid_lines: bool = False,
    ) -> np.ndarray:
        """
        Render coloured congestion cells over the frame.
        """
        H, W = frame.shape[:2]
        cell_h = H / self.grid_rows
        cell_w = W / self.grid_cols
        overlay = frame.copy()
        level_map = self.get_level_map(smoothed)

        colors = [
            CONGESTION_LEVELS["free"]["color_bgr"],
            CONGESTION_LEVELS["moderate"]["color_bgr"],
            CONGESTION_LEVELS["heavy"]["color_bgr"],
            CONGESTION_LEVELS["severe"]["color_bgr"],
        ]

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                lvl = level_map[r, c]
                if lvl == 0:
                    continue
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), colors[lvl], -1)

        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        if draw_grid_lines:
            for r in range(1, self.grid_rows):
                y = int(r * cell_h)
                cv2.line(result, (0, y), (W, y), (80, 80, 80), 1)
            for c in range(1, self.grid_cols):
                x = int(c * cell_w)
                cv2.line(result, (x, 0), (x, H), (80, 80, 80), 1)

        return result

    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """Add a congestion level legend to bottom-left of the frame."""
        H, W = frame.shape[:2]
        items = list(CONGESTION_LEVELS.items())
        box_w, box_h = 120, 18
        padding = 6
        start_y = H - (len(items) * (box_h + padding)) - padding

        for i, (key, info) in enumerate(items):
            y = start_y + i * (box_h + padding)
            cv2.rectangle(frame, (padding, y), (padding + box_w, y + box_h),
                          info["color_bgr"], -1)
            cv2.putText(frame, info["label"],
                        (padding + 4, y + box_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    # ── Statistics ────────────────────────────────────────────────────────────
    def get_frame_stats(self, smoothed: np.ndarray, n_vehicles: int) -> Dict:
        """Statistics for the current frame."""
        level_map = self.get_level_map(smoothed)
        counts = np.bincount(level_map.ravel(), minlength=4)
        total_cells = self.grid_rows * self.grid_cols

        return {
            "frame":          self.frame_count,
            "n_vehicles":     n_vehicles,
            "pct_free":       100 * counts[0] / total_cells,
            "pct_moderate":   100 * counts[1] / total_cells,
            "pct_heavy":      100 * counts[2] / total_cells,
            "pct_severe":     100 * counts[3] / total_cells,
            "congestion_idx": float(np.mean(level_map)),   # 0–3 index
        }

    def get_summary(self) -> Dict:
        """Aggregate statistics over all processed frames."""
        if not self.total_vehicles:
            return {}

        avg_count = np.mean(self.total_vehicles)
        peak_frame = int(np.argmax(self.total_vehicles))
        avg_zone = self.zone_totals / max(self.frame_count, 1)

        # Hotspot cells (top 5% by mean count)
        flat = avg_zone.ravel()
        threshold = np.percentile(flat, 95)
        hotspot_mask = avg_zone >= threshold

        return {
            "total_frames":     self.frame_count,
            "avg_vehicles":     float(avg_count),
            "peak_vehicles":    int(max(self.total_vehicles)),
            "peak_frame":       peak_frame,
            "hotspot_cells":    int(hotspot_mask.sum()),
            "avg_zone_counts":  avg_zone,
        }

    def draw_heatmap_cumulative(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate a full-resolution cumulative heatmap from zone totals.
        Useful for generating a 'traffic hotspot' map over a whole video.
        """
        H, W = frame_shape
        avg_zone = self.zone_totals / max(self.frame_count, 1)
        norm = avg_zone / avg_zone.max() if avg_zone.max() > 0 else avg_zone
        upsampled = cv2.resize(norm.astype(np.float32), (W, H),
                               interpolation=cv2.INTER_CUBIC)
        heat_uint8 = (upsampled * 255).astype(np.uint8)
        return cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)


def load_congestion_mapper_from_config(config_path: str = "config.yaml") -> CongestionMapper:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ccfg = cfg["congestion"]
    return CongestionMapper(
        grid_rows=ccfg["grid_rows"],
        grid_cols=ccfg["grid_cols"],
        thresholds=ccfg["thresholds"],
        smoothing_sigma=ccfg["smoothing_sigma"],
    )
