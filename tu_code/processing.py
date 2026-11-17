from io_utils import normalize_to_u8
import numpy as np

def compute_magnitude(fx, fy):
    M = np.hypot(fx.astype(np.float64), fy.astype(np.float64))
    M_u8 = normalize_to_u8(M)
    return M, M_u8


def compute_gradient_direction(fx, fy):
    phi = np.arctan2(fy.astype(np.float64), fx.astype(np.float64))
    phi_deg = np.degrees(phi)
    phi_deg = (phi_deg + 360.0) % 360.0
    phi_img = np.uint8((phi_deg / 360.0) * 255.0)
    return phi_img, phi_deg


def quantize_gradient_direction(phi_degrees):
    q = np.zeros_like(phi_degrees, dtype=np.uint8)
    angle = phi_degrees % 360.0
    q[((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle < 202.5)) | ((angle >= 337.5) & (angle < 360))] = 0
    q[((angle >= 22.5) & (angle < 67.5)) | ((angle >= 202.5) & (angle < 247.5))] = 1
    q[((angle >= 67.5) & (angle < 112.5)) | ((angle >= 247.5) & (angle < 292.5))] = 2
    q[((angle >= 112.5) & (angle < 157.5)) | ((angle >= 292.5) & (angle < 337.5))] = 3
    return q


def non_maxima_suppression(M_float, quantized_dirs):
    """Keep maxima along the gradient; y increases down the image."""
    suppressed = np.zeros_like(M_float, dtype=np.float64)
    center = M_float[1:-1, 1:-1]
    directions = quantized_dirs[1:-1, 1:-1]
    pairs = [
        (M_float[1:-1, :-2], M_float[1:-1, 2:]),
        (M_float[:-2, :-2], M_float[2:, 2:]),
        (M_float[:-2, 1:-1], M_float[2:, 1:-1]),
        (M_float[:-2, 2:], M_float[2:, :-2]),
    ]
    for direction, (n1, n2) in enumerate(pairs):
        keep = (directions == direction) & (center >= n1) & (center >= n2)
        suppressed[1:-1, 1:-1][keep] = center[keep]
    suppressed_u8 = normalize_to_u8(suppressed)
    return suppressed, suppressed_u8


def hysteresis_thresholding(suppressed_float, Th, Tl):
    if not (np.isfinite(Th) and np.isfinite(Tl) and 0 < Tl <= Th <= 255):
        raise ValueError("Require 0 < low threshold <= high threshold <= 255")
    s_min, s_max = suppressed_float.min(), suppressed_float.max()
    if s_max != s_min:
        suppressed_vis = ((suppressed_float - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
    else:
        suppressed_vis = np.zeros_like(suppressed_float, dtype=np.uint8)

    rows, cols = suppressed_vis.shape
    suppressed_vis[0, :] = 0
    suppressed_vis[-1, :] = 0
    suppressed_vis[:, 0] = 0
    suppressed_vis[:, -1] = 0

    edges = np.zeros((rows, cols), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=bool)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def follow(r0, c0):
        stack = [(r0, c0)]
        visited[r0, c0] = True
        while stack:
            r, c = stack.pop()
            edges[r, c] = 255
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (not visited[nr, nc]) and (suppressed_vis[nr, nc] >= Tl):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if suppressed_vis[r, c] >= Th and not visited[r, c]:
                follow(r, c)

    return edges

