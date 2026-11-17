"""Binary 3 x 3 morphology, implemented with shifted NumPy arrays."""

import numpy as np


def _morph(mask, dilate):
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    result = np.zeros_like(mask) if dilate else np.ones_like(mask)
    height, width = mask.shape
    for row in range(3):
        for col in range(3):
            view = padded[row:row + height, col:col + width]
            if dilate:
                result |= view
            else:
                result &= view
    return result


def dilate(mask, iterations=1):
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        result = _morph(result, True)
    return result


def erode(mask, iterations=1):
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        result = _morph(result, False)
    return result


def close(mask, iterations=1):
    return erode(dilate(mask, iterations), iterations)
