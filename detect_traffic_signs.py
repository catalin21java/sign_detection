"""
Traffic road signs detection — reads configuration from input.txt next to this script.

Detects sign-like regions via HSV color masks (red / blue / yellow) and contour analysis.
Supports diamond / rhombus warning signs (yellow inner field, common in EU) using
extent vs. axis-aligned bbox and min-area rectangle aspect.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_TXT = SCRIPT_DIR / "input.txt"


def parse_input_txt(path: Path) -> tuple[Path | None, Path, list[Path]]:
    """
    Parse input.txt:
    - images_dir: <folder>   — process all *.jpg, *.jpeg, *.png, *.bmp in folder
    - image: <file> or bare lines ending in image extensions
    Returns (single explicit dir or None, output_dir, list of explicit image paths).
    """
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8", errors="replace")
    output_dir = SCRIPT_DIR / "output"
    images_dir: Path | None = None
    explicit_paths: list[Path] = []
    ext_pattern = re.compile(r"\.(jpe?g|png|bmp|webp)$", re.IGNORECASE)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("output_dir:") or low.startswith("output:"):
            output_dir = (SCRIPT_DIR / line.split(":", 1)[1].strip()).resolve()
            continue
        if low.startswith("images_dir:") or low.startswith("input_dir:") or low.startswith("folder:"):
            images_dir = (SCRIPT_DIR / line.split(":", 1)[1].strip()).resolve()
            continue
        if low.startswith("image:") or low.startswith("file:"):
            explicit_paths.append((SCRIPT_DIR / line.split(":", 1)[1].strip()).resolve())
            continue
        if ext_pattern.search(line) and ":" not in line.split()[0]:
            explicit_paths.append((SCRIPT_DIR / line).resolve())

    return images_dir, output_dir, explicit_paths


def collect_image_paths(images_dir: Path | None, explicit: list[Path]) -> list[Path]:
    files: list[Path] = []
    globs = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    if images_dir and images_dir.is_dir():
        for g in globs:
            files.extend(sorted(images_dir.glob(g)))
    for p in explicit:
        if p.is_file():
            files.append(p)
    # unique, stable order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _hsv_masks_raw(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Red, blue, yellow (strict), yellow (wide for shadowed panels) as separate masks."""
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 70, 50])
    upper2 = np.array([180, 255, 255])
    red = cv2.inRange(hsv, lower1, upper1)
    red = cv2.bitwise_or(red, cv2.inRange(hsv, lower2, upper2))

    blue = cv2.inRange(hsv, np.array([100, 80, 60]), np.array([130, 255, 255]))

    yellow = cv2.inRange(hsv, np.array([18, 70, 70]), np.array([38, 255, 255]))
    # Broader yellow for shaded sign faces (lower S/V).
    yellow_wide = cv2.inRange(hsv, np.array([12, 35, 45]), np.array([42, 255, 255]))

    yellow_combined = cv2.bitwise_or(yellow, yellow_wide)
    return red, blue, yellow_combined, yellow_wide


def hsv_masks_from_hsv(hsv: np.ndarray) -> np.ndarray:
    """Combined sign-color mask with morphology (pass precomputed BGR→HSV)."""
    red, blue, yel, _ = _hsv_masks_raw(hsv)
    combined = cv2.bitwise_or(red, blue)
    combined = cv2.bitwise_or(combined, yel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined


def hsv_masks(img: np.ndarray) -> np.ndarray:
    return hsv_masks_from_hsv(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))


def _extent_and_minrect(cnt: np.ndarray) -> tuple[float, float]:
    """extent = area / axis_bbox_area; min_ar = min(w,h)/max(w,h) of min-area rectangle."""
    area = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect_a = float(bw * bh)
    extent = area / rect_a if rect_a > 1e-6 else 0.0
    (_, (mw, mh), _) = cv2.minAreaRect(cnt)
    mw, mh = float(mw), float(mh)
    if max(mw, mh) < 1e-6:
        return extent, 0.0
    min_ar = min(mw, mh) / max(mw, mh)
    return extent, min_ar


def classify_shape(cnt: np.ndarray) -> str:
    """
    Shape label including rhombus (romb / diamond road sign: rotated square).
    Uses polygon vertex count, fill ratio inside axis-aligned bbox, and min-area rect aspect.
    """
    peri = cv2.arcLength(cnt, True)
    if peri < 1e-6:
        return "unknown"
    area = cv2.contourArea(cnt)
    extent, min_ar = _extent_and_minrect(cnt)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    v = len(approx)
    circ = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0.0

    # Diamond / rhombus panel: much lower extent than a filled rectangle (≈0.5 ideal).
    if 0.32 < extent < 0.82 and min_ar > 0.62:
        if v == 4 or (0.40 < extent < 0.72 and min_ar > 0.72):
            return "rhombus"

    if v == 3:
        return "triangular"
    if v == 4:
        if extent > 0.78:
            return "rectangular"
        if min_ar > 0.72:
            return "rhombus"
        return "quadrilateral"
    if v >= 6 and circ > 0.65:
        return "circular"
    if circ > 0.75:
        return "circular"
    return "sign_region"


def dominant_color_name(img_bgr: np.ndarray, cnt: np.ndarray) -> str:
    """Median H/S/V inside filled contour (handles yellow / red wrap / blue)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_, w_ = hsv.shape[:2]
    mask = np.zeros((h_, w_), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    pts = hsv[mask > 0]
    if pts.size == 0:
        return "unknown"
    mh = float(np.median(pts[:, 0]))
    ms = float(np.median(pts[:, 1]))
    mv = float(np.median(pts[:, 2]))
    if ms < 45 and mv > 145:
        return "white"
    if ms < 35:
        return "gray"
    if (mh <= 12 or mh >= 158) and ms >= 45:
        return "red"
    if 90 <= mh <= 132 and ms >= 45:
        return "blue"
    if 15 <= mh <= 45 and ms >= 35:
        return "yellow"
    if 35 < mh < 85 and ms >= 45:
        return "green"
    if mh < 90 and ms >= 45:
        return "orange"
    return "mixed"


def contour_solidity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    if area < 1e-6:
        return 0.0
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    return area / ha if ha > 1e-6 else 0.0


def spectral_sign_fractions(hsv: np.ndarray, cnt: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    Hue vote among saturated pixels inside contour (mutually exclusive bins).
    Returns (f_yellow, f_red, f_blue, f_green) or None if too few pixels.
    """
    h_, w_ = hsv.shape[:2]
    mask = np.zeros((h_, w_), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    pts = hsv[mask > 0]
    if pts.shape[0] < 48:
        return None
    good = pts[:, 1] >= 38
    pts = pts[good]
    if pts.shape[0] < 40:
        return None
    hh = pts[:, 0].astype(np.int32)
    n = float(pts.shape[0])

    # One label per hue (avoid double-counting olive leaves as both yellow + green).
    is_red = (hh <= 14) | (hh >= 157)
    is_yellow = (hh >= 15) & (hh <= 50)
    is_blue = (hh >= 92) & (hh <= 136)
    is_green = (hh >= 55) & (hh <= 93)

    fy = np.count_nonzero(is_yellow) / n
    fr = np.count_nonzero(is_red) / n
    fb = np.count_nonzero(is_blue) / n
    fg = np.count_nonzero(is_green) / n

    return (float(fy), float(fr), float(fb), float(fg))


def iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 1e-6 else 0.0


def nms_quad_rects(rects_with_meta: list[dict], iou_thresh: float = 0.22) -> list[dict]:
    """Prefer larger blobs; suppress near-duplicates overlapping the same patch."""
    rects_with_meta.sort(key=lambda d: cv2.contourArea(d["cnt"]), reverse=True)
    kept: list[dict] = []
    for d in rects_with_meta:
        r = d["rect"]
        if any(iou_xywh(r, k["rect"]) > iou_thresh for k in kept):
            continue
        kept.append(d)
    return kept


def is_sign_colored_histogram(
    fy: float, fr: float, fb: float, fg: float, *, shape: str
) -> bool:
    """Suppress foliage where green dominates; keep clear sign hues."""
    best = max(fy, fr, fb)
    if best < 0.33:
        return False
    if fg > 0.52 and best < 0.45:
        return False
    if fg > best + 0.08 and shape not in ("rhombus", "rectangular"):
        return False
    # Yellow rhombus (romb): allow moderate green bleed from background in bbox
    if shape == "rhombus" and fy >= 0.28 and fg <= 0.55:
        return True
    return best >= 0.36 or (shape == "rhombus" and fy >= 0.30)


def process_image(path: Path, out_dir: Path) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return

    h, w = img.shape[:2]
    min_area = max(520, (h * w) // 3800)
    max_area = (h * w) * 0.35

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red, blue, yel, _ = _hsv_masks_raw(hsv)
    kernel_y = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    yellow_focus = cv2.morphologyEx(yel, cv2.MORPH_CLOSE, kernel_y, iterations=3)
    yellow_focus = cv2.morphologyEx(yellow_focus, cv2.MORPH_OPEN, kernel_y, iterations=1)

    mask_all = hsv_masks_from_hsv(hsv)
    mask = cv2.bitwise_or(mask_all, yellow_focus)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh) if bh > 0 else 0
        if aspect < 0.25 or aspect > 4.0:
            continue
        min_side = min(bw, bh)
        if min_side < 22:
            continue

        sol = contour_solidity(cnt)
        if sol < 0.66:
            continue

        shape = classify_shape(cnt)
        specs = spectral_sign_fractions(hsv, cnt)
        if specs is None:
            continue
        fy, fr, fb, fg = specs
        if not is_sign_colored_histogram(fy, fr, fb, fg, shape=shape):
            continue
        if shape == "sign_region" and max(fy, fr, fb) < 0.42:
            continue
        if shape == "sign_region" and min_side < 52 and cv2.contourArea(cnt) < 12000:
            continue

        color_name = dominant_color_name(img, cnt)
        shape_disp = "romb" if shape == "rhombus" else shape
        candidates.append(
            {
                "rect": (x, y, bw, bh),
                "cnt": cnt,
                "label": f"{color_name} | {shape_disp}",
                "area_px": cv2.contourArea(cnt),
            }
        )

    kept = sorted(
        nms_quad_rects(candidates, iou_thresh=0.22),
        key=lambda d: d["area_px"],
        reverse=True,
    )
    print(f"{path.name} - {len(kept)} detection(s)")
    print(f"  saved: {(out_dir / f'marked_{path.name}').resolve()}")
    for i, item in enumerate(kept, start=1):
        x, y, bw, bh = item["rect"]
        px = int(round(item["area_px"]))
        print(f"  [{i}] {item['label']} - bbox ({x},{y}) size {bw}x{bh} px - area ~{px} px")
    vis = img.copy()
    box_color = (0, 255, 0)
    for item in kept:
        x, y, bw, bh = item["rect"]
        label = item["label"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), box_color, 2)
        cv2.putText(
            vis,
            label,
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"marked_{path.name}"
    cv2.imwrite(str(out_path), vis)


def main() -> None:
    images_dir, output_dir, explicit = parse_input_txt(INPUT_TXT)
    paths = collect_image_paths(images_dir, explicit)
    if not paths:
        print(
            "No images found. Add to input.txt e.g.\n"
            "  images_dir: images\n"
            "or list files: mysigns/photo1.jpg",
            file=sys.stderr,
        )
        sys.exit(1)
    for p in paths:
        process_image(p, output_dir)


if __name__ == "__main__":
    main()
