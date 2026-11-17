"""Convert Canny boundaries to one filled outer silhouette per object.

The background uses 4-connectivity, complementary to 8-connected edges.
Filled silhouettes deliberately include holes. Original closed edges are kept
separately so that feature extraction can recover interior regions.
"""

from collections import deque
from dataclasses import dataclass
import numpy as np
from morphology import close

BORDER_MARGIN = 3     # cach mep < 3 px coi la cham mep (Canny luon bo vien 1 px)
MIN_BBOX_FILL = 0.08  # vung chiem < 8% khung bao la soi bien ho, khong phai vat


@dataclass
class Component:
    id: int
    bbox: tuple  # (x0, y0, x1, y1), upper bounds excluded
    mask: np.ndarray  # local filled silhouette inside bbox


def connected_components(mask, connectivity=8):
    """Return a label image and pixel coordinate arrays, using BFS."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = mask.shape
    regions = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if labels[row, col]:
            continue
        label = len(regions) + 1
        queue = deque([(row, col)])
        labels[row, col] = label
        pixels = []
        while queue:
            r, c = queue.popleft()
            pixels.append((r, c))
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width
                        and mask[nr, nc] and labels[nr, nc] == 0):
                    labels[nr, nc] = label
                    queue.append((nr, nc))
        regions.append(np.asarray(pixels, dtype=np.int32))
    return labels, regions


def fill_enclosed_regions(edges):
    """Flood exterior background with scanline spans (4-connected)."""
    edges = np.asarray(edges, dtype=bool)
    height, width = edges.shape
    exterior = np.zeros_like(edges)
    seeds = ([(0, c) for c in range(width)] + [(height - 1, c) for c in range(width)]
             + [(r, 0) for r in range(height)] + [(r, width - 1) for r in range(height)])
    while seeds:
        row, col = seeds.pop()
        if edges[row, col] or exterior[row, col]:
            continue
        left = right = col
        while left > 0 and not edges[row, left - 1] and not exterior[row, left - 1]:
            left -= 1
        while right + 1 < width and not edges[row, right + 1] and not exterior[row, right + 1]:
            right += 1
        exterior[row, left:right + 1] = True
        for nr in (row - 1, row + 1):
            if not 0 <= nr < height:
                continue
            free = ~edges[nr, left:right + 1] & ~exterior[nr, left:right + 1]
            starts = free & ~np.r_[False, free[:-1]]
            seeds.extend((nr, left + int(offset)) for offset in np.flatnonzero(starts))
    return ~exterior


def segment_objects(edges, config):
    closed = close(np.asarray(edges) > 0, config.closing_iterations)
    filled = fill_enclosed_regions(closed)
    raw_labels, regions = connected_components(filled, 8)
    labels = np.zeros_like(raw_labels)
    components = []
    warnings = []
    height, width = filled.shape
    for index, pixels in enumerate(regions, 1):
        if len(pixels) < config.min_area:
            continue
        y0, x0 = pixels.min(axis=0).tolist()
        y1, x1 = (pixels.max(axis=0) + 1).tolist()
        if (x0 < BORDER_MARGIN or y0 < BORDER_MARGIN
                or x1 > width - BORDER_MARGIN or y1 > height - BORDER_MARGIN):
            warnings.append("Rejected a component touching the image border")
            continue
        if len(pixels) < MIN_BBOX_FILL * (x1 - x0) * (y1 - y0):
            warnings.append("Rejected a thin open-edge fragment")
            continue
        if len(pixels) > config.max_area_fraction * filled.size:
            warnings.append("Rejected an oversized component; inspect broken/merged edges")
            continue
        object_id = len(components) + 1
        local = raw_labels[y0:y1, x0:x1] == index
        labels[y0:y1, x0:x1][local] = object_id
        components.append(Component(object_id, (x0, y0, x1, y1), local))
    return components, labels, closed, warnings
