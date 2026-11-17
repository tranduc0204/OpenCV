"""
Dem & phan loai linh kien bang thu vien OpenCV - di DUNG CAC BUOC nhu ban tu code (tu_code/).
Buoc nao OpenCV co ham san thi dung ham do. Luat phan loai (classifier.py) va tham so
(default.json) dung CHUNG voi ban tu code -> khac biet ket qua chi do phan cai dat.

  Buoc                     Ban tu code (tu_code/)          Ban nay (OpenCV)
  1. Doc anh, anh xam      Pillow + nhan ma tran RGB       cv2.imdecode + cv2.cvtColor
  2. Canny                 canny.py (NumPy)                cv2.GaussianBlur + cv2.Canny
  3. Dong bien             morphology.close                cv2.morphologyEx(MORPH_CLOSE)
  4. Loang nen             fill_enclosed_regions           cv2.floodFill
  5. Tach vat              connected_components (BFS)      cv2.connectedComponentsWithStats
  6. Tam, do thuon         ma tran hiep phuong sai         cv2.moments
  7. Bien ngoai vat        morphology.erode                cv2.erode
  8. Tim lo                connected_components (BFS)      cv2.connectedComponentsWithStats
  9. radial_cv, hex_score  cong thuc rieng                 giu nguyen (OpenCV khong co ham tuong ung)
 10. Phan loai             classifier.classify             classifier.classify (dung chung)

Cach dung:
    pip install opencv-python numpy
    python test_opencv.py anh.jpg            # chay 1 anh
    python test_opencv.py dataXuLiAnh/       # chay ca thu muc
Ket qua luu trong thu muc "output/" (canh file nay):
    <ten>_steps.png   : anh tung buoc (goc, Canny, dong bien, vat the, lo, ket qua)
    <ten>_result.png  : anh ket qua co nhan
    objects.csv       : dac trung tung vat
    counts.csv        : so luong tung loai moi anh
Neu anh co trong tu_code/counts.json (nhan dem tay) thi in them so dem that va do chinh xac.
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TU_CODE = HERE / "tu_code"
sys.path.insert(0, str(TU_CODE))
from classifier import CLASS_NAMES, classify  # noqa: E402  (luat dung chung voi ban tu code)
from config import load_config  # noqa: E402

CONFIG_PATH = TU_CODE / "default.json"
LABELS_PATH = TU_CODE / "counts.json"
OUTDIR = HERE / "output"
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
FONT = cv2.FONT_HERSHEY_SIMPLEX
BORDER_MARGIN = 3     # giong segmentation.py: cach mep < 3 px coi la cham mep
MIN_BBOX_FILL = 0.08  # giong segmentation.py: chiem < 8% khung bao la soi bien ho
COLORS = {"bolt": (220, 105, 38), "washer": (92, 147, 22),      # BGR, cung mau voi ban tu code
          "nut": (20, 117, 215), "unknown": (75, 47, 195)}
NAMES = {"bolt": "Bu-long", "washer": "Vong dem", "nut": "Dai oc", "unknown": "Chua xac dinh"}
FEATURE_KEYS = ("outer_area", "material_area_estimate", "elongation", "radial_cv",
                "hex_score", "hole_count", "hole_area_fraction")


# ---------------------------------------------------------------- doc / ghi anh
def read_image(path):
    # imdecode doc duoc duong dan co dau tieng Viet; bo qua EXIF xoay anh giong Pillow
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {path} (kiem tra co phai .jpg/.png)")
    return img


def write_image(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise ValueError(f"Khong ghi duoc anh: {path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------- buoc 2: Canny
def gaussian_size(sigma, tail):
    """Cung cong thuc kich thuoc bo loc voi filters.calculate_filter_size."""
    half = int(round(np.sqrt(-np.log(tail) * 2.0 * sigma ** 2)))
    return 2 * half + 1


def detect_edges(gray, config):
    size = gaussian_size(config.sigma, config.gaussian_tail)
    blur = cv2.GaussianBlur(gray, (size, size), config.sigma, borderType=cv2.BORDER_REPLICATE)
    # Ban tu code dat nguong tren thang 0-255 chuan hoa theo gradient lon nhat cua anh;
    # cv2.Canny dung don vi gradient tuyet doi -> quy doi: nguong = T / 255 * gradient_lon_nhat.
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    peak = float(cv2.magnitude(gx, gy).max())
    low = config.low_threshold / 255.0 * peak
    high = config.high_threshold / 255.0 * peak
    return cv2.Canny(blur, low, high, apertureSize=3, L2gradient=True)


# ---------------------------------------------------------------- buoc 3-5: tach vat
def segment_objects(edges, config):
    if config.closing_iterations > 0:
        # vien = 0 o ca dilate lan erode, giong morphology.close cua ban tu code
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, KERNEL,
                                  iterations=config.closing_iterations,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        closed = edges.copy()

    # Loang nen 4-lien thong tu mep anh (them vien 1 px de moi diem mep deu noi voi (0,0)).
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 128, flags=4)
    filled = (padded[1:-1, 1:-1] != 128).astype(np.uint8)   # vat the da lap kin ca lo

    n, raw, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    height, width = filled.shape
    labels = np.zeros(raw.shape, np.int32)
    components = []
    for index in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < config.min_area:
            continue
        if (x < BORDER_MARGIN or y < BORDER_MARGIN
                or x + w > width - BORDER_MARGIN or y + h > height - BORDER_MARGIN):
            continue                                   # cham / sat mep anh
        if area < MIN_BBOX_FILL * w * h:
            continue                                   # soi bien ho, khong phai vat
        if area > config.max_area_fraction * filled.size:
            continue                                   # qua lon: bien dut / dinh nhau
        object_id = len(components) + 1
        local = raw[y:y + h, x:x + w] == index
        labels[y:y + h, x:x + w][local] = object_id
        components.append({"id": object_id, "bbox": (x, y, x + w, y + h),
                           "mask": local, "area": area})
    return components, labels, closed > 0


# ---------------------------------------------------------------- buoc 6-9: dac trung
def radial_shape(mask, center, eigenvalues, eigenvectors):
    """Cong thuc giong features._radial_shape; chi thay erode bang cv2.erode."""
    eroded = cv2.erode(mask.astype(np.uint8), KERNEL,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    boundary = mask & ~eroded
    points = np.argwhere(boundary).astype(float) - center
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


def find_holes(component, gray, closed, config, center):
    """Cung luat voi features._find_holes; tach vung ben trong bang cv2."""
    x0, y0, x1, y1 = component["bbox"]
    silhouette = component["mask"]
    area = component["area"]
    local_gray = gray[y0:y1, x0:x1]
    inside = (silhouette & ~closed[y0:y1, x0:x1]).astype(np.uint8)
    n, lab, stats, centroids = cv2.connectedComponentsWithStats(inside, connectivity=4)
    # Do sang nen lay tu vanh hep quanh khung bao cua vat
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
    count = 0
    for k in range(1, n):
        fraction = stats[k, cv2.CC_STAT_AREA] / area
        if not config.min_hole_fraction <= fraction <= config.max_hole_fraction:
            continue
        candidate_center = np.array([centroids[k][1], centroids[k][0]])   # (hang, cot)
        if np.linalg.norm(candidate_center - center) > 0.35 * np.sqrt(area):
            continue
        region = lab == k
        intensity = float(np.median(local_gray[region]))
        if contrast < 5 or abs(intensity - background) > config.hole_background_tolerance * contrast:
            continue
        hole_mask |= region
        count += 1
    return count, hole_mask


def extract_features(component, gray, closed, config):
    mask = component["mask"]
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    area = m["m00"]
    center = np.array([m["m01"] / area, m["m10"] / area])              # (hang, cot)
    covariance = np.array([[m["mu02"], m["mu11"]], [m["mu11"], m["mu20"]]]) / area
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    elongation = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-8)))
    radial_cv, hex_score = radial_shape(mask, center, eigenvalues, eigenvectors)
    hole_count, hole_mask = find_holes(component, gray, closed, config, center)
    x0, y0, _, _ = component["bbox"]
    features = {
        "outer_area": int(area), "material_area_estimate": int(area) - int(hole_mask.sum()),
        "centroid": [float(center[1] + x0), float(center[0] + y0)],
        "elongation": elongation, "radial_cv": radial_cv, "hex_score": hex_score,
        "hole_count": hole_count, "hole_area_fraction": float(hole_mask.sum() / area),
    }
    return features, hole_mask


# ---------------------------------------------------------------- ca pipeline
def analyze_image(path, config):
    start = time.perf_counter()
    img = read_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray, config)
    components, labels, closed = segment_objects(edges, config)
    counts = {name: 0 for name in CLASS_NAMES}
    objects = []
    holes = np.zeros(gray.shape, dtype=bool)
    for component in components:
        features, hole_mask = extract_features(component, gray, closed, config)
        label, reason = classify(features, config)
        counts[label] += 1
        objects.append({"id": component["id"], "label": label, "bbox": list(component["bbox"]),
                        "features": features, "reason": reason})
        x0, y0, x1, y1 = component["bbox"]
        holes[y0:y1, x0:x1] |= hole_mask
    report = {"image": str(path), "counts": counts, "total": len(objects), "objects": objects,
              "processing_seconds": time.perf_counter() - start}
    stages = {"bgr": img, "edges": edges, "closed": closed, "labels": labels, "holes": holes}
    return report, stages


# ---------------------------------------------------------------- ve ket qua
def draw_result(img, report):
    out = img.copy()
    height, width = out.shape[:2]
    scale = max(0.6, min(height, width) / 900)
    thick = max(2, int(round(scale * 2)))
    for obj in report["objects"]:
        x0, y0, x1, y1 = obj["bbox"]
        color = COLORS[obj["label"]]
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, thick)
        text = f"#{obj['id']} {NAMES[obj['label']]}"
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        left = max(0, min(x0, width - tw - 10))
        top = max(0, y0 - th - base - 10)
        cv2.rectangle(out, (left, top), (left + tw + 10, top + th + base + 8), color, -1)
        cv2.putText(out, text, (left + 5, top + th + 4), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def steps_grid(stages, report, result, tile_width=640):
    lut = np.full((stages["labels"].max() + 1, 3), 245, np.uint8)
    for obj in report["objects"]:
        lut[obj["id"]] = COLORS[obj["label"]]
    tiles = [("1. Anh goc", stages["bgr"]), ("2. cv2.Canny", stages["edges"]),
             ("3. Dong bien", stages["closed"].astype(np.uint8) * 255),
             ("4. Vat the", lut[stages["labels"]]),
             ("5. Lo", stages["holes"].astype(np.uint8) * 255), ("6. Ket qua", result)]
    out = []
    for title, tile in tiles:
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        h, w = tile.shape[:2]
        tile = cv2.resize(tile, (tile_width, int(round(h * tile_width / w))), interpolation=cv2.INTER_AREA)
        (tw, th), base = cv2.getTextSize(title, FONT, 0.8, 2)
        cv2.rectangle(tile, (0, 0), (tw + 16, th + base + 12), (255, 255, 255), -1)
        cv2.putText(tile, title, (8, th + 6), FONT, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
        out.append(tile)
    return cv2.vconcat([cv2.hconcat(out[:3]), cv2.hconcat(out[3:])])


# ---------------------------------------------------------------- chay tu dong lenh
def natural_key(path):
    digits = "".join(filter(str.isdigit, path.stem))
    return (int(digits) if digits else 0, path.name)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    src = Path(sys.argv[1])
    if src.is_dir():
        paths = sorted((p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                       key=natural_key)
    else:
        paths = [src]
    config = load_config(CONFIG_PATH)
    truth = {}
    if LABELS_PATH.exists():
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8-sig"))
        truth = {e["image"]: e["counts"] for e in labels["images"]}
    OUTDIR.mkdir(exist_ok=True)

    rows = []
    with (OUTDIR / "objects.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["anh", "id", "loai", "x0", "y0", "x1", "y1", "cx", "cy", *FEATURE_KEYS, "ly_do"])
        for path in paths:
            report, stages = analyze_image(path, config)
            result = draw_result(stages["bgr"], report)
            write_image(OUTDIR / f"{path.stem}_result.png", result)
            write_image(OUTDIR / f"{path.stem}_steps.png", steps_grid(stages, report, result))
            for obj in report["objects"]:
                f = obj["features"]
                writer.writerow([path.name, obj["id"], obj["label"], *obj["bbox"],
                                 round(f["centroid"][0], 1), round(f["centroid"][1], 1),
                                 *(round(f[k], 4) if isinstance(f[k], float) else f[k] for k in FEATURE_KEYS),
                                 obj["reason"]])
            counts, ms = report["counts"], report["processing_seconds"] * 1000
            real = truth.get(path.name)
            ok = None
            line = f"{path.name}: {counts}  ({ms:.0f} ms)"
            if real:
                ok = all(counts[k] == real[k] for k in real) and counts["unknown"] == 0
                line += f"  that={real}  {'DUNG' if ok else 'SAI'}"
            print(line, flush=True)
            rows.append((path.name, counts, ms, real, ok))

    with (OUTDIR / "counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["anh", *CLASS_NAMES, "thoi_gian_ms", "that_bolt", "that_washer", "that_nut", "dung"])
        for name, counts, ms, real, ok in rows:
            real = real or {}
            writer.writerow([name, *(counts[k] for k in CLASS_NAMES), round(ms, 1),
                             real.get("bolt", ""), real.get("washer", ""), real.get("nut", ""),
                             "" if ok is None else int(ok)])

    judged = [ok for *_, ok in rows if ok is not None]
    if judged:
        print(f"\nDem dung hoan toan: {sum(judged)}/{len(judged)} anh ({100 * sum(judged) / len(judged):.1f}%)")
    print(f"Da luu ket qua vao {OUTDIR}")


if __name__ == "__main__":
    main()
