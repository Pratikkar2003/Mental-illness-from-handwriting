"""
feature_extraction.py
----------------------
Turns a photo/scan of handwriting into the kind of feature representation
the EMOTHAW-style pipeline in Khan et al. (2024) uses.

IMPORTANT HONESTY NOTE
=======================
The original paper analyzes *online* handwriting captured live on a
digitizing tablet: for every point it has x, y, timestamp, pen-up/down
status, azimuth, altitude and PRESSURE. That is what lets it compute real
kinematic/dynamic features (velocity, pen-lift patterns, true pressure).

A photo of finished handwriting is *offline/static* data: no timestamps,
no true pressure, no pen-lift information. This module extracts the best
available PROXIES from the image instead:

    - stroke geometry (a resampled point sequence per connected component,
      standing in for "stroke points")
    - local stroke width (proxy for pressure - harder press -> thicker/darker line)
    - local ink darkness (second pressure proxy)
    - local curvature / direction change (proxy for dynamic changes in movement)
    - slant angle per component
    - inter-word/inter-letter spacing, size variance, baseline drift
      (classic offline graphology-style features used in the emotion/
      handwriting literature)

These proxies are reasonable and used in real offline-handwriting research,
but they are NOT the same signal as the paper's tablet data, and any model
trained on them will not automatically reach the paper's accuracy. Train on
your own labeled data for numbers you can trust.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

MAX_SEQ_LEN = 200          # fixed length stroke-point sequence fed to the transformer
POINT_FEATURE_DIM = 6      # x, y, local_width, local_darkness, curvature, slant
GLOBAL_FEATURE_DIM = 10    # engineered "time/frequency/statistical domain" style features


@dataclass
class HandwritingFeatures:
    point_sequence: np.ndarray   # (MAX_SEQ_LEN, POINT_FEATURE_DIM)
    seq_mask: np.ndarray         # (MAX_SEQ_LEN,) 1 = real point, 0 = padding
    global_features: np.ndarray  # (GLOBAL_FEATURE_DIM,)


def _load_binary_ink_mask(image: np.ndarray) -> np.ndarray:
    """Grayscale + adaptive threshold -> binary mask where ink = 255."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Otsu thresholding is robust to lighting variation in scans/photos
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # remove tiny speckle noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary, gray


def _component_slant_angle(contour: np.ndarray) -> float:
    """Estimate slant of a stroke component using PCA on its pixel coordinates."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return 0.0
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(principal[1], principal[0]))
    # normalize to [-90, 90], measuring deviation from vertical (90 deg = upright)
    if angle < 0:
        angle += 180
    return angle - 90.0


def _resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly n points, evenly spaced by arc length."""
    if len(points) < 2:
        return np.repeat(points, n, axis=0)[:n]
    deltas = np.diff(points, axis=0)
    seg_lens = np.sqrt((deltas ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total == 0:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0, total, n)
    out = np.zeros((n, points.shape[1]), dtype=np.float32)
    for dim in range(points.shape[1]):
        out[:, dim] = np.interp(targets, cum, points[:, dim])
    return out


def extract_features(image: np.ndarray) -> HandwritingFeatures:
    """
    Main entry point. `image` is a BGR or grayscale numpy array (e.g. from
    cv2.imread or a PIL image converted with np.array).
    Returns a HandwritingFeatures object ready to feed the model.
    """
    binary, gray = _load_binary_ink_mask(image)
    h, w = binary.shape

    # distance transform -> local stroke half-width at every ink pixel (pressure proxy 1)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 3 or len(c) > 8]

    if len(contours) == 0:
        # blank / unreadable image -> return zeroed features, caller should warn user
        return HandwritingFeatures(
            point_sequence=np.zeros((MAX_SEQ_LEN, POINT_FEATURE_DIM), dtype=np.float32),
            seq_mask=np.zeros((MAX_SEQ_LEN,), dtype=np.float32),
            global_features=np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32),
        )

    # order components in natural reading order: top-to-bottom rows, then left-to-right
    boxes = [cv2.boundingRect(c) for c in contours]
    row_key = lambda b: (round(b[1] / max(h * 0.03, 1)), b[0])
    order = sorted(range(len(contours)), key=lambda i: row_key(boxes[i]))

    all_points = []
    widths, darkness_vals, slants, heights, gaps, centers_x = [], [], [], [], [], []
    prev_right_edge = None

    for idx in order:
        c = contours[idx]
        x, y, bw, bh = boxes[idx]
        pts = c.reshape(-1, 2).astype(np.float32)

        # local width / darkness sampled along the stroke
        local_widths = []
        local_dark = []
        for (px, py) in pts[:: max(1, len(pts) // 40)]:
            xi, yi = int(px), int(py)
            if 0 <= yi < h and 0 <= xi < w:
                local_widths.append(dist[yi, xi])
                local_dark.append(255 - gray[yi, xi])
        widths.extend(local_widths if local_widths else [0.0])
        darkness_vals.extend(local_dark if local_dark else [0.0])

        slant = _component_slant_angle(c)
        slants.append(slant)
        heights.append(bh)
        centers_x.append(x + bw / 2.0)
        if prev_right_edge is not None:
            gaps.append(max(0.0, x - prev_right_edge))
        prev_right_edge = x + bw

        # normalized (x, y) trace of this component, resampled to a modest length
        n_pts = max(4, min(20, len(pts)))
        resampled = _resample_polyline(pts, n_pts)
        norm = resampled.copy()
        norm[:, 0] = norm[:, 0] / max(w, 1)
        norm[:, 1] = norm[:, 1] / max(h, 1)

        # curvature proxy: local direction change along the resampled trace
        curv = np.zeros(len(resampled), dtype=np.float32)
        if len(resampled) >= 3:
            v1 = resampled[1:-1] - resampled[:-2]
            v2 = resampled[2:] - resampled[1:-1]
            n1 = np.linalg.norm(v1, axis=1) + 1e-6
            n2 = np.linalg.norm(v2, axis=1) + 1e-6
            cosang = np.clip((v1 * v2).sum(axis=1) / (n1 * n2), -1, 1)
            curv[1:-1] = np.arccos(cosang)

        local_w_norm = np.interp(
            np.linspace(0, 1, len(resampled)),
            np.linspace(0, 1, max(1, len(local_widths))),
            local_widths if local_widths else [0.0],
        ) / (dist.max() + 1e-6)
        local_d_norm = np.interp(
            np.linspace(0, 1, len(resampled)),
            np.linspace(0, 1, max(1, len(local_dark))),
            local_dark if local_dark else [0.0],
        ) / 255.0

        for i in range(len(resampled)):
            all_points.append([
                norm[i, 0], norm[i, 1],
                float(local_w_norm[i]), float(local_d_norm[i]),
                float(curv[i] / np.pi), float(slant / 90.0),
            ])

    all_points = np.array(all_points, dtype=np.float32)
    if len(all_points) == 0:
        point_seq = np.zeros((MAX_SEQ_LEN, POINT_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((MAX_SEQ_LEN,), dtype=np.float32)
    elif len(all_points) >= MAX_SEQ_LEN:
        idxs = np.linspace(0, len(all_points) - 1, MAX_SEQ_LEN).astype(int)
        point_seq = all_points[idxs]
        mask = np.ones((MAX_SEQ_LEN,), dtype=np.float32)
    else:
        pad = np.zeros((MAX_SEQ_LEN - len(all_points), POINT_FEATURE_DIM), dtype=np.float32)
        point_seq = np.concatenate([all_points, pad], axis=0)
        mask = np.concatenate([
            np.ones((len(all_points),), dtype=np.float32),
            np.zeros((MAX_SEQ_LEN - len(all_points),), dtype=np.float32),
        ])

    # ---- global "time/frequency/statistical domain" style summary features ----
    # NOTE: raw pixel measurements (stroke width, letter height, etc.) scale
    # directly with the input image's resolution/DPI. A phone photo can be
    # 10-50x more pixels than a small scan, which would otherwise make two
    # visually-identical handwriting samples look totally different to the
    # model. We normalize width/darkness by the sample's own average letter
    # height so features describe *relative* stroke thickness (e.g. "stroke
    # width is 8% of letter height") instead of an absolute pixel count.
    heights_arr = np.array(heights, dtype=np.float32) if heights else np.array([0.0])
    scale = float(heights_arr.mean()) + 1e-6

    widths_arr = (np.array(widths, dtype=np.float32) if widths else np.array([0.0])) / scale
    dark_arr = (np.array(darkness_vals, dtype=np.float32) if darkness_vals else np.array([0.0])) / 255.0
    slants_arr = np.array(slants, dtype=np.float32) if slants else np.array([0.0])
    gaps_arr = np.array(gaps, dtype=np.float32) if gaps else np.array([0.0])

    def spectral_entropy(sig: np.ndarray) -> float:
        if len(sig) < 4:
            return 0.0
        f = np.fft.rfft(sig - sig.mean())
        p = np.abs(f) ** 2
        p = p / (p.sum() + 1e-9)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def autocorr1(sig: np.ndarray) -> float:
        if len(sig) < 2:
            return 0.0
        s = sig - sig.mean()
        denom = (s ** 2).sum() + 1e-9
        return float((s[:-1] * s[1:]).sum() / denom)

    global_features = np.array([
        widths_arr.mean(), widths_arr.std(),
        dark_arr.mean(), dark_arr.std(),
        slants_arr.mean(), slants_arr.std(),
        heights_arr.std() / (heights_arr.mean() + 1e-6),   # size irregularity
        gaps_arr.std() / (gaps_arr.mean() + 1e-6) if len(gaps_arr) > 1 else 0.0,  # spacing irregularity
        spectral_entropy(widths_arr),
        autocorr1(widths_arr),
    ], dtype=np.float32)
    # replace any nan/inf from degenerate images
    global_features = np.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0)

    return HandwritingFeatures(point_sequence=point_seq, seq_mask=mask, global_features=global_features)


def extract_features_from_path(path: str) -> HandwritingFeatures:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return extract_features(image)
