"""E-N1 (H4): native-scale synthetic sheets with real glyph rendering.

Calibration/eval ground truth at REAL absolute scale (measured from the real
scans this session): ruling lines 28-60 px; glyph blocks 150-190 px with
25-60 px strokes and compact bodies; frame double-bands; AA edges; JPEG
ringing; noise. Masks are exact by construction.

No "clean reference" is emitted: metrics work on (photo, result, masks)
against a constant paper level with a control (identity) arm for the noise
floor.

Files per window: {stem}_photo.png, {stem}_line_mask.png (line cores incl.
halo), {stem}_char_mask.png (true glyph/punct ink), {stem}_meta.json.

Usage:
  .venv/bin/python evidence/methods/gen_synth.py \
      --out data/native_synth/dev --n 12 --seed 1000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
]
GLYPHS = "一二三十大小天人日月口中田由甲申回火水木金土王玉石馬鳥魚龍花鸟"
GLYPHS += "風雨山川气上中下左右内外出入开门関雪雲雷电田時分年千百万元"

_font_cache = {}


def get_font(size: int, cache=_font_cache):
    if size not in cache:
        for path in FONT_CANDIDATES:
            try:
                cache[size] = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("no CJK font available")
    return cache[size]


def draw_line_layer(img, rng, nprng):
    """Vertical ruling lines + top/bottom frame double-bands.
    Returns list of corridor dicts (x0,x1 inclusive-exclusive spans)."""
    h, w = img.shape
    corridors = []
    n_v = rng.randint(3, 5)
    xs = sorted(rng.sample(range(300, w - 300), n_v))
    for x in xs:
        width = rng.randint(28, 60)
        core = rng.randint(80, 180)
        x0 = x - width // 2
        x1 = x0 + width
        breaks = []
        y = rng.randint(50, 300)
        while y < h - 200:
            if rng.random() < 0.35:
                g = rng.randint(30, 250)
                breaks.append((int(y), int(y + g)))
                y += g
            y += rng.randint(400, 900)
        for b in range(max(0, x0), min(w, x1)):
            col = np.full(h, float(core)) + nprng.normal(0, 8, h)
            for g0, g1 in breaks:
                col[g0:g1] = 255.0
            # halo: extra 6 px each side, partial darkness
            img[:, b] = np.minimum(img[:, b], col)
        for side in (x0 - 6, x1):
            s0, s1 = max(0, side), max(0, side) + 6
            if s1 > s0:
                col_h = np.full(h, float(core) + 50) \
                    + nprng.normal(0, 8, h)
                for g0, g1 in breaks:
                    col_h[g0:g1] = 255.0
                for b in range(s0, min(w, s1)):
                    img[:, b] = np.minimum(img[:, b], col_h)
        corridors.append({"orient": "v", "x0": int(x0), "x1": int(x1),
                          "y0": 0, "y1": int(h), "center": int(x),
                          "core": int(core), "breaks": breaks})
    for y in (rng.randint(60, 180), h - rng.randint(60, 180)):
        w1 = rng.randint(18, 34)
        gap = 12 + rng.randint(0, 20)
        w2 = rng.randint(10, 18)
        y0 = max(0, y)
        band = np.zeros(h, np.float32)
        v1 = rng.randint(20, 60)
        v2 = rng.randint(60, 110)
        img[y0:y0 + w1, :] = np.minimum(img[y0:y0 + w1, :], v1)
        img[y0 + w1 + gap:y0 + w1 + gap + w2, :] = np.minimum(
            img[y0 + w1 + gap:y0 + w1 + gap + w2, :], v2)
        corridors.append({"orient": "h", "x0": 0, "x1": int(w),
                          "y0": int(y0), "y1": int(y0 + w1 + gap + w2),
                          "center": int(y0 + w1), "core": 30,
                          "breaks": []})
    return corridors


def render_glyph(img, mask_char, gx, gy, size, kind, rng, nprng):
    h, w = img.shape
    if kind == "glyph":
        font = get_font(size)
        ch = rng.choice(GLYPHS)
        x0, y0 = max(0, gx - 12), max(0, gy - 12)
        x1, y1 = min(w, gx + size + 12), min(h, gy + size + 12)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return
        patch = Image.new("L", (x1 - x0, y1 - y0), 255)
        d = ImageDraw.Draw(patch)
        d.text((gx - x0, gy - y0), ch, font=font, fill=0)
        sub = np.asarray(patch) < 200
    else:
        sub = np.zeros((min(size, h - gy), min(size // 2, w - gx)), bool)
        sub[:] = True
        x0, y0 = gx, gy
        x1, y1 = min(w, gx + size // 2), min(h, gy + size)
    dark_val = rng.randint(15, 50)
    region = img[y0:y1, x0:x1]
    vals = dark_val + nprng.normal(0, 6, int(sub.sum()))
    region[sub] = np.minimum(region[sub], vals)
    img[y0:y1, x0:x1] = region
    sub_full = np.zeros(img.shape, bool)
    sub_full[y0:y1, x0:x1] = sub
    mask_char |= sub_full


def draw_glyphs(img, mask_char, corridors, rng, nprng, hard: bool):
    h, w = img.shape
    vxs = sorted(set(c["center"] for c in corridors if c["orient"] == "v"))
    bounds = [40] + vxs + [w - 40]
    placed = []
    for i in range(len(bounds) - 1):
        lx, rx = bounds[i], bounds[i + 1]
        if rx - lx < 200:
            continue
        y = rng.randint(250, 400)
        last_size = 170
        while y < h - 300:
            n_in_row = max(1, (rx - lx) // 210)
            for _ in range(n_in_row):
                kind = "punct" if rng.random() < 0.10 else "glyph"
                size = rng.randint(150, 190) if kind == "glyph" else \
                    rng.randint(18, 26)
                gx = rng.randint(max(41, lx + 5),
                                 max(42, min(w - size - 5, rx - size - 5)))
                gy = int(max(250, y + rng.randint(-10, 10)))
                render_glyph(img, mask_char, gx, gy, size, kind, rng, nprng)
                placed.append([gx, gy, size, kind])
            last_size = size
            y += last_size + rng.randint(30, 90)
    # planted intersections: 1-2 glyphs re-rendered straddling a line
    if hard and placed and vxs:
        for _ in range(rng.randint(1, 2)):
            gx, gy, size, kind = rng.choice([p for p in placed
                                             if p[3] == "glyph"])
            line = rng.choice(vxs)
            render_glyph(img, mask_char, int(line - size / 2),
                         int(gy + rng.randint(-40, 40)), size, "glyph",
                         rng, nprng)
    return placed


def inject_hard(img, char_mask, corridors, rng, nprng):
    """Replicate the THREE real residual-failure ingredients measured on the
    real scans: dark-paper zones (flank gates fail), stroke-scale ink
    traveling ALONG the corridors (protection/traffic cases), punctuation-
    scale blobs sitting ON the lines (must never be removed)."""
    h, w = img.shape
    # 1) dark-paper blotches: smooth depressions of 20..45 grey
    for _ in range(rng.randint(3, 6)):
        cx, cy = rng.randint(100, w - 100), rng.randint(100, h - 100)
        rad = rng.randint(150, 500)
        blob = np.zeros((h, w), np.uint8)
        cv2.circle(blob, (cx, cy), rad, 255, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), rad / 3)
        dep = rng.uniform(20, 45) * (blob / 255.0).astype(np.float32)
        img -= dep
    vcorrs = [c for c in corridors if c["orient"] == "v"]
    for c in vcorrs:
        # 2) stroke-scale proxies along the line
        for _ in range(rng.randint(1, 3)):
            side = rng.choice((-1, 1))
            x = c["center"] + side * ((c["x1"] - c["x0"]) // 2
                                       + rng.randint(2, 8))
            sw = rng.randint(60, 90)
            x0 = x if side > 0 else x - sw
            x0 = max(0, min(w - sw, x0))
            y0 = rng.randint(300, h - 1000)
            ln = rng.randint(300, 900)
            dark = rng.randint(30, 60)
            blk = dark + nprng.normal(0, 5, (ln, sw))
            region = img[y0:y0 + ln, x0:x0 + sw]
            region[:] = np.minimum(region, blk)
            img[y0:y0 + ln, x0:x0 + sw] = region
            char_mask[y0:y0 + ln, x0:x0 + sw] = 1
        # 3) punctuation-like blobs ON the line
        for _ in range(rng.randint(1, 3)):
            cy = rng.randint(300, h - 400)
            s = rng.randint(18, 26)
            x0 = max(0, c["center"] - s // 2)
            region = img[cy:cy + s, x0:x0 + s]
            region[:] = rng.randint(20, 60)
            img[cy:cy + s, x0:x0 + s] = region
            char_mask[cy:cy + s, x0:x0 + s] = 1


def make_window(seed: int, hard: bool):
    rng = random.Random(seed)
    prng = np.random.default_rng(seed)
    w, h = 2000, 2800
    # paper
    img = np.full((h, w), 245.0, np.float32)
    tex = cv2.GaussianBlur(prng.normal(0, 3.0, (h, w)).astype(np.float32),
                           (0, 0), 2.0)
    img += tex
    # lines
    corridors = draw_line_layer(img, rng, prng)
    line_mask = np.zeros((h, w), np.uint8)
    for c in corridors:
        if c["orient"] == "v":
            line_mask[:, max(0, c["x0"] - 6):c["x1"] + 6] = 1
        else:
            line_mask[c["y0"]:c["y1"], :] = 1
    char_mask = np.zeros((h, w), np.uint8)
    draw_glyphs(img, char_mask, corridors, rng, prng, hard)
    if hard:
        inject_hard(img, char_mask, corridors, rng, prng)
    # char_mask should not credit halo: intersect with actual dark ink
    char_mask &= (img < 200)
    # degrade
    img = np.clip(img + prng.normal(0, rng.uniform(1.5, 4.0), (h, w)),
                  0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        import io
        q = rng.randint(75, 88)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = np.asarray(Image.open(buf).convert("L"))
    return img, line_mask, char_mask, corridors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        seed = args.seed + i
        hard = i % 3 == 2
        photo, line_mask, char_mask, corridors = make_window(seed, hard)
        stem = f"n{seed:05d}"
        cv2.imwrite(str(out / f"{stem}_photo.png"), photo)
        cv2.imwrite(str(out / f"{stem}_line_mask.png"), line_mask * 255)
        cv2.imwrite(str(out / f"{stem}_char_mask.png"), char_mask * 255)
        (out / f"{stem}_meta.json").write_text(json.dumps(
            {"seed": seed, "hard": hard, "corridors": corridors}))
        print(stem, "h" if hard else "-",
              f"line_px={int(line_mask.sum())}",
              f"char_px={int(char_mask.sum())}", flush=True)


if __name__ == "__main__":
    main()