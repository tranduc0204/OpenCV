"""Application pipeline: image -> Canny -> objects -> features -> labels -> counts."""

from pathlib import Path
from time import perf_counter
import csv
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from canny import detect_edges
from classifier import CLASS_NAMES, DISPLAY_NAMES, classify
from config import Config
from features import extract_features
from segmentation import segment_objects

COLORS = {"bolt": (38, 105, 220), "washer": (22, 147, 92),
          "nut": (215, 117, 20), "unknown": (195, 47, 75)}


def analyze_image(image_path, config=None):
    config = config or Config()
    start = perf_counter()
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    # Manual RGB -> gray conversion; Pillow is used only for image I/O/drawing.
    gray = np.rint(rgb.astype(float) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    edges, stages = detect_edges(gray, config)
    components, labels, closed, warnings = segment_objects(edges, config)
    objects = []
    counts = {name: 0 for name in CLASS_NAMES}
    holes = np.zeros(gray.shape, dtype=bool)
    for component in components:
        features, hole_mask = extract_features(component, gray, closed, config)
        label, reason = classify(features, config)
        counts[label] += 1
        objects.append({"id": component.id, "label": label,
                        "display_name": DISPLAY_NAMES[label], "bbox": list(component.bbox),
                        "features": features, "reason": reason})
        x0, y0, x1, y1 = component.bbox
        holes[y0:y1, x0:x1] |= hole_mask
    if not components:
        warnings.append("No objects found; inspect edge closure and minimum area")
    if counts["unknown"]:
        warnings.append("Some objects are unknown; inspect holes and shape measurements")
    label_rgb = np.full(rgb.shape, 245, dtype=np.uint8)
    for obj in objects:
        label_rgb[labels == obj["id"]] = COLORS[obj["label"]]
    stages.update({"rgb": rgb, "closed": closed.astype(np.uint8) * 255,
                   "labels": label_rgb, "holes": holes.astype(np.uint8) * 255})
    report = {"image": str(Path(image_path).resolve()),
              "image_size": [gray.shape[1], gray.shape[0]],
              "parameters": config.to_dict(), "counts": counts,
              "total": len(objects), "objects": objects, "warnings": warnings,
              "processing_seconds": round(perf_counter() - start, 4)}
    return report, stages


def _font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_overlay(rgb, report):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    font = _font(max(14, min(image.size) // 45))
    line_width = max(2, min(image.size) // 300)
    for obj in report["objects"]:
        x0, y0, x1, y1 = obj["bbox"]
        color = COLORS[obj["label"]]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=line_width)
        label = f"#{obj['id']} {obj['display_name']}"
        text_box = draw.textbbox((0, 0), label, font=font)
        tw, th = text_box[2] - text_box[0] + 10, text_box[3] - text_box[1] + 10
        left = max(0, min(x0, image.width - tw))
        top = max(0, y0 - th - 2)
        draw.rectangle((left, top, left + tw, top + th), fill=color)
        draw.text((left + 5, top + 3 - text_box[1]), label, fill="white", font=font)
    return image


def save_result(report, stages, output_dir, overview=True):
    output_dir = Path(output_dir)
    stage_dir = output_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name, array in stages.items():
        if name != "rgb":
            Image.fromarray(array).save(stage_dir / f"{name}.png")
    overlay = render_overlay(stages["rgb"], report)
    overlay.save(output_dir / "overlay.png")
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_names = list(report["objects"][0]["features"]) if report["objects"] else []
    with (output_dir / "objects.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "label", "bbox", *feature_names, "reason"])
        writer.writeheader()
        for obj in report["objects"]:
            writer.writerow({"id": obj["id"], "label": obj["label"], "bbox": obj["bbox"],
                             **obj["features"], "reason": obj["reason"]})
    if overview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(12, 9))
        panels = [(stages["rgb"], "Original"), (stages["edges"], "Manual Canny"),
                  (stages["closed"], "Closed edges"), (stages["labels"], "Object silhouettes"),
                  (stages["holes"], "Hole candidates"), (np.asarray(overlay), "Classification")]
        for ax, (array, title) in zip(axes.flat, panels):
            ax.imshow(array, cmap="gray" if array.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(" | ".join(f"{name}: {value}" for name, value in report["counts"].items()))
        fig.tight_layout()
        fig.savefig(output_dir / "overview.png", dpi=140)
        plt.close(fig)
    return output_dir
