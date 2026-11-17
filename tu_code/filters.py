import numpy as np

def calculate_filter_size(sigma, T):
    if sigma < 0.5:
        raise ValueError("sigma must be >= 0.5")
    if not (0.0 < T < 1.0):
        raise ValueError("T must be in (0,1)")
    sHalf = int(round(np.sqrt(-np.log(T) * 2.0 * (sigma**2))))
    N = 2 * sHalf + 1
    return N, sHalf


def calculate_gradient(filter_size, sigma, scale_factor=255):
    sHalf = filter_size // 2
    coords = np.arange(-sHalf, sHalf + 1)
    # Image coordinates: x increases along columns, y along rows.
    X, Y = np.meshgrid(coords, coords)
    G = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    Gx = -(X / (sigma**2)) * G
    Gy = -(Y / (sigma**2)) * G
    Gx_int = np.rint(Gx * scale_factor).astype(np.int32)
    Gy_int = np.rint(Gy * scale_factor).astype(np.int32)
    return Gx_int, Gy_int, scale_factor


def convolve(image, kernel):
    """Apply the existing correlation convention using only NumPy.

    Both derivative kernels have the same global sign reversal compared with
    mathematical convolution; edge directions modulo 180 degrees are unchanged.
    Loop over kernel entries rather than individual image pixels.
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image.astype(np.int64), ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros((img_h, img_w), dtype=np.int64)
    for i in range(k_h):
        for j in range(k_w):
            result += padded[i:i+img_h, j:j+img_w] * int(kernel[i, j])
    return result


def apply_masks(image, Gx_int, Gy_int, scale_factor):
    fx_scaled = convolve(image.astype(np.int32), Gx_int)
    fy_scaled = convolve(image.astype(np.int32), Gy_int)
    fx = np.rint(fx_scaled.astype(np.float64) / float(scale_factor)).astype(np.int32)
    fy = np.rint(fy_scaled.astype(np.float64) / float(scale_factor)).astype(np.int32)
    return fx, fy

