# -*- coding: utf-8 -*-
"""
cell_morph_features.py

Compute per-cell traditional morphology features from an instance-labeled segmentation mask.

Assumptions:
- mask: 2D array [H,W], integer labels. Background = 0.
- For each cell id in `cell_ids`, the corresponding instance pixels satisfy (mask == cell_id).
  If a cell_id is missing in the mask, its feature vector will be zeros.

Features kept (per user request):
- area
- bbox_area
- convex_area
- eccentricity
- equivalent_diameter
- extent
- major_axis_length
- minor_axis_length
- orientation
- perimeter
- solidity
- hw_ratios  (major/minor)

NOTE:
- We intentionally DO NOT include filled_area (not meaningful for cells here).

Implementation:
- Uses skimage.measure.regionprops for robust, standard definitions.
- Optionally standardizes features across cells (z-score), with log1p for size-like features.

Outputs:
- feat_np: float32 [N,K]
- names: list[str] length K (order matches columns)

"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Dict, Any

import numpy as np

try:
    from skimage.measure import regionprops
except Exception as e:
    raise ImportError("scikit-image is required for morphology feature extraction (skimage.measure.regionprops).") from e


# -----------------------------
# helpers
# -----------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _bbox_area_from_bbox(bbox: Sequence[int]) -> float:
    # bbox = (min_row, min_col, max_row, max_col)
    r0, c0, r1, c1 = bbox
    return float(max(0, r1 - r0) * max(0, c1 - c0))


def _equiv_diameter_from_area(a: float) -> float:
    # diameter of circle with same area
    a = max(0.0, float(a))
    return float(2.0 * np.sqrt(a / (np.pi + 1e-12)))


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def compute_cell_morph_features(
    mask: np.ndarray,
    cell_ids: Sequence[int] | Sequence[str],
    *,
    standardize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Args:
      mask: [H,W] instance labels (0 background)
      cell_ids: length N list, corresponding to meta/data cell ids
      standardize: if True, log1p size-like features then z-score across cells

    Returns:
      feat_np: [N,K] float32
      names: list of feature names (length K)
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D [H,W], got shape={mask.shape}")

    # normalize ids -> int
    ids: List[int] = []
    for cid in cell_ids:
        try:
            ids.append(int(cid))
        except Exception:
            # if cannot cast, keep placeholder -1; will end up zeros
            ids.append(-1)

    N = len(ids)

    # compute regionprops for all labels present (exclude background)
    # regionprops expects non-negative ints
    m = np.asarray(mask)
    if m.dtype.kind not in ("i", "u"):
        m = m.astype(np.int32, copy=False)

    props = regionprops(m)

    # map label -> props
    prop_map: Dict[int, Any] = {int(p.label): p for p in props}

    # feature names (order fixed!)
    names = [
        "area",
        "bbox_area",
        "convex_area",
        "eccentricity",
        "equivalent_diameter",
        "extent",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "perimeter",
        "solidity",
        "hw_ratios",
    ]
    K = len(names)
    feat = np.zeros((N, K), dtype=np.float32)

    for i, cid in enumerate(ids):
        p = prop_map.get(int(cid), None)
        if p is None:
            continue

        area = _safe_float(getattr(p, "area", 0.0), 0.0)
        bbox_area = _safe_float(getattr(p, "area_bbox", None), None)
        if bbox_area is None:
            bbox_area = _bbox_area_from_bbox(getattr(p, "bbox", (0, 0, 0, 0)))
        convex_area = _safe_float(getattr(p, "area_convex", None), None)
        if convex_area is None:
            # fallback: area of convex_image
            try:
                convex_area = float(np.sum(getattr(p, "convex_image")))
            except Exception:
                convex_area = 0.0

        eccentricity = _safe_float(getattr(p, "eccentricity", 0.0), 0.0)
        equiv_d = _safe_float(getattr(p, "equivalent_diameter_area", None), None)
        if equiv_d is None:
            equiv_d = _equiv_diameter_from_area(area)

        extent = _safe_float(getattr(p, "extent", 0.0), 0.0)
        major = _safe_float(getattr(p, "major_axis_length", 0.0), 0.0)
        minor = _safe_float(getattr(p, "minor_axis_length", 0.0), 0.0)
        orientation = _safe_float(getattr(p, "orientation", 0.0), 0.0)
        perimeter = _safe_float(getattr(p, "perimeter", 0.0), 0.0)
        solidity = _safe_float(getattr(p, "solidity", 0.0), 0.0)
        hw = float(major / (minor + 1e-6))

        feat[i, :] = np.array(
            [
                area,
                bbox_area,
                convex_area,
                eccentricity,
                equiv_d,
                extent,
                major,
                minor,
                orientation,
                perimeter,
                solidity,
                hw,
            ],
            dtype=np.float32,
        )

    if standardize:
        # log1p for size-like features to reduce heavy-tailed scale
        # (area / bbox_area / convex_area / diameter / major/minor/perimeter)
        size_idx = [0, 1, 2, 4, 6, 7, 9]
        feat2 = feat.copy()
        feat2[:, size_idx] = np.log1p(np.clip(feat2[:, size_idx], a_min=0.0, a_max=None))
        feat2 = _zscore(feat2)
        feat = feat2.astype(np.float32, copy=False)

    return feat.astype(np.float32, copy=False), names
