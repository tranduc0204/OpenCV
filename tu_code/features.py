"""Rotation-insensitive shape measurements, implemented with NumPy.

Outer silhouettes drive object counting; enclosed Canny regions and local
background intensity together provide an approximate hole measurement.
"""

import numpy as np
from morphology import erode
from segmentation import connected_components


def _radial_shape(mask, center, eigenvalues, eigenvectors):
    boundary = mask & ~erode(mask)
    points = np.argwhere(boundary).astype(float) - center
    # Whitening removes ellipse distortion while preserving hexagonal variation.
    points = (points @ eigenvectors) / np.sqrt(np.maximum(eigenvalues, 1e-8))
    angles = np.mod(np.arctan2(points[:, 0], points[:, 1]), 2 * np.pi)
    distances = np.linalg.norm(points, axis=1)
    bins = 180
    indices = np.minimum((angles * bins / (2 * np.pi)).astype(int), bins - 1)
    radius = np.zeros(bins, dtype=float)
    np.maximum.at(radius, indices, distances)
    known = np.flatnonzero(radius > 0)
    if len(known) < 12:
        return 1.0, 0.0
    radius = np.interp(np.arange(bins), np.r_[known - bins, known, known + bins],
                       np.tile(radius[known], 3))
    radius /= radius.mean()
    theta = (np.arange(bins) + 0.5) * (2 * np.pi / bins)
    radial_cv = float(np.std(radius))
    hex_score = float(2 * abs(np.mean((radius - 1) * np.exp(-6j * theta))))
    return radial_cv, hex_score


def _find_holes(component, gray, closed_edges, config):
    x0, y0, x1, y1 = component.bbox
    silhouette = component.mask
    local_gray = gray[y0:y1, x0:x1]
    local_edges = closed_edges[y0:y1, x0:x1]
    _, regions = connected_components(silhouette & ~local_edges, 4)
    # Reference background from a narrow ring around this object's bounding box.
    margin = 6
    xa, ya = max(0, x0 - margin), max(0, y0 - margin)
    xb, yb = min(gray.shape[1], x1 + margin), min(gray.shape[0], y1 + margin)
    patch = gray[ya:yb, xa:xb]
    ring = np.ones(patch.shape, dtype=bool)
    ring[y0 - ya:y1 - ya, x0 - xa:x1 - xa] = False
    background = float(np.median(patch[ring]))
    material = float(np.median(local_gray[silhouette]))
    contrast = abs(background - material)
    hole_mask = np.zeros_like(silhouette)
    center = np.argwhere(silhouette).mean(axis=0)
    area = int(silhouette.sum())
    count = 0
    for pixels in regions:
        fraction = len(pixels) / area
        if not config.min_hole_fraction <= fraction <= config.max_hole_fraction:
            continue
        candidate_center = pixels.mean(axis=0)
        if np.linalg.norm(candidate_center - center) > 0.35 * np.sqrt(area):
            continue
        intensity = float(np.median(local_gray[pixels[:, 0], pixels[:, 1]]))
        if contrast < 5 or abs(intensity - background) > config.hole_background_tolerance * contrast:
            continue
        hole_mask[pixels[:, 0], pixels[:, 1]] = True
        count += 1
    return count, hole_mask


def extract_features(component, gray, closed_edges, config):
    mask = component.mask
    pixels = np.argwhere(mask).astype(float)
    center = pixels.mean(axis=0)
    centered = pixels - center
    covariance = centered.T @ centered / len(pixels)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    elongation = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-8)))
    radial_cv, hex_score = _radial_shape(mask, center, eigenvalues, eigenvectors)
    hole_count, hole_mask = _find_holes(component, gray, closed_edges, config)
    # Four-neighbour exposed-edge perimeter is a diagnostic lattice estimate.
    padded = np.pad(mask.astype(np.int8), 1)
    perimeter = int(np.abs(np.diff(padded, axis=0)).sum()
                    + np.abs(np.diff(padded, axis=1)).sum())
    area = len(pixels)
    x0, y0, _, _ = component.bbox
    features = {
        "outer_area": area, "material_area_estimate": area - int(hole_mask.sum()),
        "centroid": [float(center[1] + x0), float(center[0] + y0)],
        "elongation": elongation, "outer_perimeter_grid": perimeter,
        "circularity_grid": float(4 * np.pi * area / max(perimeter ** 2, 1)),
        "radial_cv": radial_cv, "hex_score": hex_score,
        "hole_count": hole_count, "hole_area_fraction": float(hole_mask.sum() / area),
    }
    return features, hole_mask
