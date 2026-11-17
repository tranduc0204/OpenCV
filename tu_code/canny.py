"""Reusable Canny pipeline, without OpenCV or other image-processing methods."""

import numpy as np
from config import Config
from filters import calculate_filter_size, calculate_gradient, apply_masks
from processing import (compute_magnitude, compute_gradient_direction,
                        quantize_gradient_direction, non_maxima_suppression,
                        hysteresis_thresholding)
from io_utils import normalize_to_u8


def detect_edges(gray, config=None):
    config = config or Config()
    gray = np.asarray(gray)
    if gray.ndim != 2 or min(gray.shape) < 5:
        raise ValueError("Expected a 2D grayscale image, at least 5 x 5 pixels")
    size, _ = calculate_filter_size(config.sigma, config.gaussian_tail)
    gx, gy, scale = calculate_gradient(size, config.sigma)
    fx, fy = apply_masks(gray, gx, gy, scale)
    magnitude, magnitude_u8 = compute_magnitude(fx, fy)
    _, angles = compute_gradient_direction(fx, fy)
    quantized = quantize_gradient_direction(angles)
    suppressed, suppressed_u8 = non_maxima_suppression(magnitude, quantized)
    edges = hysteresis_thresholding(suppressed, config.high_threshold, config.low_threshold)
    return edges, {
        "gray": gray.astype(np.uint8), "gradient_x": normalize_to_u8(fx),
        "gradient_y": normalize_to_u8(fy), "magnitude": magnitude_u8,
        "directions": quantized * 85, "nms": suppressed_u8, "edges": edges,
    }
