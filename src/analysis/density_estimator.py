"""
src/analysis/density_estimator.py
──────────────────────────────────
Traffic density estimation from vehicle detections.

Two methods:
  1. KDE  — Kernel Density Estimation (smooth Gaussian heatmap)
  2. Grid — Divide frame into a grid, count vehicles per cell

Both return a normalized float32 density map (H×W) and a colored heatmap (H×W×3).
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional
import yaml


class KDEDensityEstimator:
    """
    Gaussian kernel density estimator for vehicle positions.

    Each detected vehicle centre contributes a Gaussian "blob" to
    the density map. Summing all blobs gives a smooth density field.
    """

    def __init__(self, bandwidth: int = 30, colormap: int = cv2.COLORMAP_JET):
        self.bandwidth = bandwidth
        self.colormap = colormap

    def estimate(
        self,
        frame_shape: Tuple[int, int],
        centers: List[Tuple[float, float]],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Compute KDE density map.

        Parameters
        ----------
        frame_shape : (H, W)
        centers : list of (x, y) pixel coordinates (vehicle centres)
        weights : optional per-vehicle weights (e.g. 1.0 for car, 2.0 for bus)

        Returns
        -------
        density : float32 array of shape (H, W), values in [0, 1]
        """
        H, W = frame_shape
        density = np.zeros((H, W), dtype=np.float32)

        if not centers:
            return density

        if weights is None:
            weights = [1.0] * len(centers)

        for (x, y), w in zip(centers, weights):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                density[yi, xi] += w

        # Smooth with Gaussian filter (bandwidth = sigma in pixels)
        density = gaussian_filter(density, sigma=self.bandwidth)

        # Normalize to [0, 1]
        max_val = density.max()
        if max_val > 0:
            density /= max_val

        return density

    def to_heatmap(
        self, density: np.ndarray, alpha: float = 0.55
    ) -> np.ndarray:
        """
        Convert density map → BGR heatmap image.

        Parameters
        ----------
        density : float32 (H, W) in [0, 1]
        alpha   : not used here; apply when overlaying on frame

        Returns
        -------
        heatmap : uint8 (H, W, 3) BGR
        """
        heat_uint8 = (density * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat_uint8, self.colormap)
        return heatmap

    def overlay(
        self,
        frame: np.ndarray,
        density: np.ndarray,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """Blend heatmap over the original frame."""
        heatmap = self.to_heatmap(density)
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)


class GridDensityEstimator:
    """
    Divides the frame into an R×C grid and counts vehicles per cell.
    Faster and more interpretable than KDE for congestion mapping.
    """

    def __init__(
        self,
        grid_cols: int = 20,
        grid_rows: int = 20,
        smoothing_sigma: float = 1.5,
        colormap: int = cv2.COLORMAP_HOT,
    ):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.sigma = smoothing_sigma
        self.colormap = colormap

    def estimate(
        self,
        frame_shape: Tuple[int, int],
        centers: List[Tuple[float, float]],
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Count vehicles per grid cell.

        Returns
        -------
        grid_counts : int array (grid_rows, grid_cols) — raw counts
        density     : float32 (H, W) — full-res density map
        """
        H, W = frame_shape
        cell_h = H / self.grid_rows
        cell_w = W / self.grid_cols
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        if weights is None:
            weights = [1.0] * len(centers)

        for (x, y), w in zip(centers, weights):
            col = min(int(x / cell_w), self.grid_cols - 1)
            row = min(int(y / cell_h), self.grid_rows - 1)
            grid[row, col] += w

        # Smooth grid
        grid_smooth = gaussian_filter(grid, sigma=self.sigma)

        # Upsample to frame size
        density = cv2.resize(
            grid_smooth, (W, H), interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)

        # Normalize
        if density.max() > 0:
            density /= density.max()

        return grid.astype(np.int32), density

    def overlay_grid(
        self,
        frame: np.ndarray,
        grid_counts: np.ndarray,
        thresholds: dict,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Draw coloured grid cells on frame based on congestion level.

        Congestion levels:
          ● free      : count < low    → green
          ● moderate  : low ≤ count < medium → yellow
          ● heavy     : medium ≤ count < high → orange
          ● severe    : count ≥ high   → red
        """
        H, W = frame.shape[:2]
        cell_h = H / self.grid_rows
        cell_w = W / self.grid_cols
        overlay = frame.copy()

        COLOR_FREE     = (0, 200, 0)       # green
        COLOR_MODERATE = (0, 220, 255)     # yellow
        COLOR_HEAVY    = (0, 120, 255)     # orange
        COLOR_SEVERE   = (0, 0, 255)       # red

        low = thresholds.get("low", 2)
        medium = thresholds.get("medium", 5)
        high = thresholds.get("high", 10)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                count = grid_counts[r, c]
                if count == 0:
                    continue

                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)

                if count < low:
                    color = COLOR_FREE
                elif count < medium:
                    color = COLOR_MODERATE
                elif count < high:
                    color = COLOR_HEAVY
                else:
                    color = COLOR_SEVERE

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.putText(
                    overlay, str(count),
                    (x1 + 2, y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
                )

        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


# ── Vehicle class weights ──────────────────────────────────────────────────────
# Larger vehicles contribute more to congestion
VEHICLE_WEIGHTS = {
    0: 1.0,   # car
    1: 1.5,   # van
    2: 2.5,   # truck
    3: 2.0,   # bus
}


def get_centers_and_weights(
    detections,   # supervision Detections object
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Extract vehicle centre coordinates and congestion weights
    from a supervision Detections object.
    """
    centers = []
    weights = []
    if detections is None or len(detections) == 0:
        return centers, weights

    for i, xyxy in enumerate(detections.xyxy):
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        w = VEHICLE_WEIGHTS.get(cls_id, 1.0)
        centers.append((cx, cy))
        weights.append(w)

    return centers, weights


def load_estimators_from_config(config_path: str = "config.yaml"):
    """Convenience factory to build estimators from project config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["density"]
    ccfg = cfg["congestion"]

    colormap_name = dcfg.get("colormap", "jet").upper()
    colormap = getattr(cv2, f"COLORMAP_{colormap_name}", cv2.COLORMAP_JET)

    kde = KDEDensityEstimator(
        bandwidth=dcfg["bandwidth"],
        colormap=colormap,
    )
    grid = GridDensityEstimator(
        grid_cols=ccfg["grid_cols"],
        grid_rows=ccfg["grid_rows"],
        smoothing_sigma=ccfg["smoothing_sigma"],
    )
    return kde, grid
