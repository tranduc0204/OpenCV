"""So sanh 2 cach dem linh kien tren cung 44 anh va cung bo nhan dem tay.

  1. tu_code : pipeline tu code (Canny + tach vat + phan loai bang NumPy), o tu_code/
  2. opencv  : test_opencv.py - cung cac buoc nhu ban tu code, buoc nao co ham OpenCV
               thi dung ham do (cv2.Canny, morphologyEx, floodFill, connectedComponents...)
Ca hai dung chung classifier.py va default.json -> khac biet chi do phan cai dat.

Them phan "do trung bien": tren CUNG mot anh xam (cong thuc cua ban tu code), so anh
bien cua Canny tu code (tu_code/canny.py) voi cv2.Canny (test_opencv.detect_edges).

tu_code/ la ban sao (chi doc) cua source_code/, configs/default.json va
data/annotations/counts.json tu project BTL. Neu sua code BTL thi chep lai.

Chay:    python so_sanh.py
Ket qua: thu muc so_sanh/
    bang_dem.csv    : so dem that / tu code / OpenCV tung anh + do trung bien tung anh
    tong_hop.json   : toan bo chi so (gom muc "do_trung_bien")
    bieu_do_dung_hoan_toan.png  : ty le anh dem dung hoan toan cua 2 ban
    bieu_do_do_trung_bien.png   : IoU va F1 (lech 1 px) trung binh giua 2 anh bien
    bieu_do_thoi_gian_canny.png : thoi gian rieng buoc Canny cua 2 ban
    bien/<anh>.png  : [bien tu code | bien OpenCV | to mau: trang = trung,
                      do = chi tu code, xanh duong = chi OpenCV] cho anh1, anh F1 thap nhat, anh40
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TU_CODE = HERE / "tu_code"
sys.path.insert(0, str(TU_CODE))
import pipeline  # noqa: E402  (pipeline tu code)
from canny import detect_edges as canny_tu_code  # noqa: E402
from config import load_config  # noqa: E402
sys.path.insert(0, str(HERE))
import test_opencv  # noqa: E402

IMAGES = HERE / "dataXuLiAnh"
LABELS = TU_CODE / "counts.json"
CONFIG = TU_CODE / "default.json"
OUT = HERE / "so_sanh"
CLASSES = ("bolt", "washer", "nut")
METHODS = ("tu_code", "opencv")
METHOD_NAMES = {"tu_code": "Tự code",
                "opencv": "OpenCV (test_opencv.py)"}
EDGE_SAMPLES = ("anh1.jpg", "anh40.jpg")   # + anh co F1 thap nhat
NEAR_3X3 = np.ones((3, 3), np.uint8)       # lech toi da 1 px theo 8 huong


def shared_gray(path):
    """Anh xam dung chung cho ca hai ban, dung cong thuc cua pipeline tu code."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return np.rint(rgb.astype(float) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


def both_edges(gray, config):
    """Tra (bien tu code, bien OpenCV, giay Canny tu code, giay cv2.Canny) dang boolean."""
    start = time.perf_counter()
    a, _ = canny_tu_code(gray, config)
    a_s = time.perf_counter() - start
    start = time.perf_counter()
    b = test_opencv.detect_edges(gray, config)
    b_s = time.perf_counter() - start
    return a > 0, b > 0, a_s, b_s


def edge_agreement(a, b):
    """A = bien tu code, B = bien OpenCV (boolean)."""
    na, nb = int(a.sum()), int(b.sum())
    union = int((a | b).sum())
    iou = (a & b).sum() / union if union else 1.0          # ca hai khong co bien -> trung hoan toan
    b_near = cv2.dilate(b.astype(np.uint8), NEAR_3X3) > 0
    a_near = cv2.dilate(a.astype(np.uint8), NEAR_3X3) > 0
    precision = (a & b_near).sum() / na if na else (1.0 if nb == 0 else 0.0)
    recall = (b & a_near).sum() / nb if nb else (1.0 if na == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"bien_tu_code_px": na, "bien_opencv_px": nb, "iou_bien": float(iou),
            "precision_lech1px": float(precision), "recall_lech1px": float(recall),
            "f1_lech1px": float(f1)}


def edge_panel(a, b, name):
    """Ghep ngang [bien tu code | bien OpenCV | to mau trung/lech], co chu ten tung o."""
    def gray3(mask):
        return cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    overlay = np.zeros((*a.shape, 3), np.uint8)
    overlay[a & b] = (255, 255, 255)   # trang: trung
    overlay[a & ~b] = (0, 0, 255)      # do (BGR): chi tu code
    overlay[b & ~a] = (255, 0, 0)      # xanh duong (BGR): chi OpenCV
    tiles = [(f"{name} - Canny tu code", gray3(a)), (f"{name} - cv2.Canny", gray3(b)),
             ("Trang = trung | Do = chi tu code | Xanh = chi OpenCV", overlay)]
    scale = max(0.6, a.shape[1] / 1000)
    thick = max(1, int(round(scale * 2)))
    out = []
    for title, tile in tiles:
        (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(tile, (0, 0), (tw + 16, th + base + 12), (60, 60, 60), -1)
        cv2.putText(tile, title, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 255), thick, cv2.LINE_AA)
        out.append(tile)
    separator = np.full((a.shape[0], 6, 3), 128, np.uint8)
    return cv2.hconcat([out[0], separator, out[1], separator, out[2]])


def run_tu_code(path, config):
    start = time.perf_counter()
    report, _ = pipeline.analyze_image(path, config)
    return dict(report["counts"]), time.perf_counter() - start


def run_opencv(path, config):
    start = time.perf_counter()
    report, _ = test_opencv.analyze_image(path, config)
    return dict(report["counts"]), time.perf_counter() - start


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def score(rows, method):
    """Chi so cua mot phuong phap so voi nhan that (cung cach cham benchmark_dataset.py)."""
    def exact(r):
        c = r[method]
        return all(c[k] == r["truth"][k] for k in CLASSES) and c["unknown"] == 0

    groups = {"tat_ca": rows,
              "roi_nhau": [r for r in rows if not r["overlap"]],
              "chong_cham_nhau": [r for r in rows if r["overlap"]]}
    result = {f"dung_hoan_toan_{name}": {"anh": sum(exact(r) for r in items), "tong": len(items),
                                         "phan_tram": 100 * sum(exact(r) for r in items) / len(items)}
              for name, items in groups.items() if items}
    result["mae_moi_loai"] = {k: float(np.mean([abs(r[method][k] - r["truth"][k]) for r in rows]))
                              for k in CLASSES}
    result["mae_tong_so"] = float(np.mean([abs(sum(r[method][k] for k in CLASSES) + r[method]["unknown"]
                                               - sum(r["truth"].values())) for r in rows]))
    result["tong_dem"] = {k: sum(r[method][k] for r in rows) for k in (*CLASSES, "unknown")}
    result["tuong_quan_pearson_voi_nhan"] = {k: pearson([r[method][k] for r in rows],
                                                        [r["truth"][k] for r in rows]) for k in CLASSES}
    result["thoi_gian_trung_binh_giay"] = float(np.mean([r["giay"][method] for r in rows]))
    result["anh_sai"] = [r["image"] for r in rows if not exact(r)]
    return result


def barh_chart(names, values, texts, xmax, ticks, tick_text, title, path, headroom=1.15):
    """Bieu do thanh ngang mot chuoi (cung kieu cho moi bieu do cua so_sanh.py).
    names/values/texts theo thu tu tu tren xuong duoi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
    fig, ax = plt.subplots(figsize=(8, 2.4), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    bars = ax.barh(names[::-1], values[::-1], height=0.5, color=series)
    for bar, text in zip(bars, texts[::-1]):
        ax.text(bar.get_width() + 0.015 * xmax, bar.get_y() + bar.get_height() / 2,
                text, va="center", color=ink, fontsize=9)
    ax.set_xlim(0, xmax * headroom)
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick_text(v) for v in ticks], color=muted, fontsize=8)
    ax.tick_params(axis="y", colors=ink2, labelsize=9, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_title(title, color=ink, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def draw_chart(summary, path):
    stats = [summary["phuong_phap"][m]["dung_hoan_toan_tat_ca"] for m in METHODS]
    barh_chart([METHOD_NAMES[m] for m in METHODS], [s["phan_tram"] for s in stats],
               [f"{s['phan_tram']:.1f}%  ({s['anh']}/{s['tong']} ảnh)" for s in stats],
               100, range(0, 101, 20), lambda v: f"{v}%",
               "Tỷ lệ ảnh đếm đúng hoàn toàn cả 3 loại (44 ảnh, nhãn đếm tay)", path)


def draw_edge_chart(summary, path):
    e = summary["do_trung_bien"]
    items = [("IoU đúng từng pixel", e["iou_dung_tung_pixel"]),
             ("F1 khi cho lệch ≤ 1 px", e["f1_lech1px"])]
    barh_chart([name for name, _ in items], [100 * s["trung_binh"] for _, s in items],
               [f"{100 * s['trung_binh']:.1f}%  (từng ảnh: {100 * s['nho_nhat']:.1f}–{100 * s['lon_nhat']:.1f}%)"
                for _, s in items],
               100, range(0, 101, 20), lambda v: f"{v}%",
               f"Độ trùng biên: Canny tự code so với cv2.Canny (trung bình {summary['so_anh']} ảnh)",
               path, headroom=1.4)


def draw_canny_time_chart(summary, path):
    t = summary["do_trung_bien"]["thoi_gian_canny_trung_binh_giay"]
    ratio = summary["do_trung_bien"]["opencv_nhanh_hon_lan"]
    values = [t["tu_code"], t["opencv"]]
    step = 10 ** np.floor(np.log10(max(values)))
    xmax = float(np.ceil(max(values) / step) * step)
    barh_chart(["Canny tự code (NumPy)", "cv2.Canny"], values,
               [f"{v:.3f} s" if v >= 0.01 else f"{v * 1000:.1f} ms" for v in values],
               xmax, np.arange(0, xmax + step / 2, step), lambda v: f"{v:g} s",
               f"Thời gian riêng bước Canny, trung bình mỗi ảnh (cv2.Canny nhanh hơn ~{ratio:.0f} lần)",
               path)


def main():
    config = load_config(CONFIG)
    labels = json.loads(LABELS.read_text(encoding="utf-8-sig"))
    truth = {e["image"]: e for e in labels["images"]}
    paths = sorted(IMAGES.glob("*.jpg"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(paths, 1):
        a_counts, a_s = run_tu_code(path, config)
        c_counts, c_s = run_opencv(path, config)
        edges_a, edges_b, canny_a_s, canny_b_s = both_edges(shared_gray(path), config)
        bien = edge_agreement(edges_a, edges_b)
        bien.update({"canny_tu_code_s": canny_a_s, "canny_opencv_s": canny_b_s})
        entry = truth[path.name]
        rows.append({"image": path.name, "truth": entry["counts"],
                     "overlap": "touching_or_overlapping" in entry.get("conditions", []),
                     "tu_code": a_counts, "opencv": c_counts,
                     "giay": {"tu_code": a_s, "opencv": c_s}, "bien": bien})
        print(f"[{i}/{len(paths)}] {path.name}  that={entry['counts']}  "
              f"tu_code={a_counts}  opencv={c_counts}  "
              f"IoU bien={bien['iou_bien']:.3f} F1(1px)={bien['f1_lech1px']:.3f}", flush=True)

    edge_rows = [r["bien"] for r in rows]
    worst = min(rows, key=lambda r: r["bien"]["f1_lech1px"])
    canny_a = float(np.mean([e["canny_tu_code_s"] for e in edge_rows]))
    canny_b = float(np.mean([e["canny_opencv_s"] for e in edge_rows]))

    def stats(key):
        values = [e[key] for e in edge_rows]
        return {"trung_binh": float(np.mean(values)), "nho_nhat": float(np.min(values)),
                "lon_nhat": float(np.max(values))}

    summary = {
        "so_anh": len(rows),
        "tham_so_chung": config.to_dict(),
        "ghi_chu": ["Hai ban dung chung classifier.py va default.json, cung cac buoc xu ly; "
                    "ban OpenCV thay moi buoc co ham san bang ham cua OpenCV.",
                    "Nguong Canny cua cv2.Canny quy doi: T / 255 * gradient lon nhat cua anh."],
        "nhan_that_tong": {k: sum(r["truth"][k] for r in rows) for k in CLASSES},
        "phuong_phap": {m: score(rows, m) for m in METHODS},
        "do_trung_bien": {
            "ghi_chu": "Cung anh xam (cong thuc ban tu code), cung default.json. "
                       "A = Canny tu code, B = cv2.Canny. Precision = ty le pixel A co B trong o 3x3; "
                       "recall = ty le pixel B co A trong o 3x3.",
            "iou_dung_tung_pixel": stats("iou_bien"),
            "f1_lech1px": stats("f1_lech1px"),
            "precision_lech1px_trung_binh": float(np.mean([e["precision_lech1px"] for e in edge_rows])),
            "recall_lech1px_trung_binh": float(np.mean([e["recall_lech1px"] for e in edge_rows])),
            "thoi_gian_canny_trung_binh_giay": {"tu_code": canny_a, "opencv": canny_b},
            "opencv_nhanh_hon_lan": canny_a / canny_b if canny_b else None,
            "anh_f1_thap_nhat": {"anh": worst["image"], **worst["bien"]},
        },
    }
    (OUT / "tong_hop.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "bang_dem.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        header = ["anh", "chong_cham"] + [f"that_{k}" for k in CLASSES]
        for m in METHODS:
            header += [f"{m}_{k}" for k in (*CLASSES, "unknown")]
        edge_cols = ["bien_tu_code_px", "bien_opencv_px", "iou_bien", "f1_lech1px",
                     "canny_tu_code_s", "canny_opencv_s"]
        writer.writerow(header + edge_cols)
        for r in rows:
            line = [r["image"], int(r["overlap"])] + [r["truth"][k] for k in CLASSES]
            for m in METHODS:
                line += [r[m][k] for k in (*CLASSES, "unknown")]
            line += [round(v, 4) if isinstance(v, float) else v for v in (r["bien"][k] for k in edge_cols)]
            writer.writerow(line)
    draw_chart(summary, OUT / "bieu_do_dung_hoan_toan.png")
    draw_edge_chart(summary, OUT / "bieu_do_do_trung_bien.png")
    draw_canny_time_chart(summary, OUT / "bieu_do_thoi_gian_canny.png")

    (OUT / "bien").mkdir(exist_ok=True)
    samples = list(dict.fromkeys([EDGE_SAMPLES[0], worst["image"], EDGE_SAMPLES[1]]))
    for name in samples:
        path = IMAGES / name
        if not path.exists():
            continue
        edges_a, edges_b, _, _ = both_edges(shared_gray(path), config)
        test_opencv.write_image(OUT / "bien" / f"{path.stem}.png", edge_panel(edges_a, edges_b, path.stem))

    print("\n=== TONG HOP ===")
    print(f"Nhan that: {summary['nhan_that_tong']}")
    for m in METHODS:
        s = summary["phuong_phap"][m]
        all_, sep = s["dung_hoan_toan_tat_ca"], s["dung_hoan_toan_roi_nhau"]
        print(f"{METHOD_NAMES[m]:24s} dung hoan toan {all_['anh']:2d}/{all_['tong']} ({all_['phan_tram']:5.1f}%)  "
              f"roi nhau {sep['anh']}/{sep['tong']}  tong dem {s['tong_dem']}  "
              f"TB {s['thoi_gian_trung_binh_giay']:.3f} s/anh")
    e = summary["do_trung_bien"]
    print(f"Do trung bien (Canny tu code vs cv2.Canny): "
          f"IoU TB {e['iou_dung_tung_pixel']['trung_binh']:.3f} "
          f"[{e['iou_dung_tung_pixel']['nho_nhat']:.3f}-{e['iou_dung_tung_pixel']['lon_nhat']:.3f}]  "
          f"F1 lech 1px TB {e['f1_lech1px']['trung_binh']:.3f} "
          f"[{e['f1_lech1px']['nho_nhat']:.3f}-{e['f1_lech1px']['lon_nhat']:.3f}]  "
          f"P {e['precision_lech1px_trung_binh']:.3f} R {e['recall_lech1px_trung_binh']:.3f}")
    print(f"Thoi gian Canny TB: tu code {canny_a:.3f} s, OpenCV {canny_b:.4f} s "
          f"(OpenCV nhanh hon {e['opencv_nhanh_hon_lan']:.0f} lan); "
          f"F1 thap nhat: {worst['image']} ({worst['bien']['f1_lech1px']:.3f})")
    print(f"\nDa luu vao {OUT}")


if __name__ == "__main__":
    main()
