"""Detect long, narrow grid boundaries on a reduced grayscale page."""
from __future__ import annotations

import cv2
import numpy as np

# working-resolution constants (tuned at 1342x2048, scale-relative intent)
TAU = 0.5          # band continuity threshold on opened-mask fraction
KERNELS = (21,)    # long-run opening kernel (work px)
VETO_W = 25        # bands wider than this cannot be ruling lines
DARK_T = 15        # darkness-above-local-background threshold (WORK scale)


def dark_mask(gu8: np.ndarray) -> np.ndarray:
    """bool mask of pixels DARKER than local background by > DARK_T.
    Only valid at working resolution (structures stay small); the
    editing and verification stages use separate brightness criteria."""
    bg = cv2.medianBlur(gu8, 31)
    return (bg.astype(np.int16) - gu8.astype(np.int16)) > DARK_T


def _line_bands(profile: np.ndarray, veto_w: int):
    """Contiguous groups with profile >= TAU, width <= veto_w."""
    active = profile >= TAU
    bands, x = [], 0
    n = len(active)
    while x < n:
        if active[x]:
            x2 = x
            while x2 < n and active[x2]:
                x2 += 1
            if x2 - x <= veto_w:
                seg = profile[x:x2]
                c = int(np.argmax(seg)) + x
                bands.append({"a": x, "b": x2, "center": c,
                              "score": float(seg.max())})
            x = x2
        else:
            x += 1
    return bands


def detect_rulings(gray: np.ndarray, work_size=(1342, 2048)):
    """Detect ruling lines at reduced resolution.

    Returns dict with:
      work:  (gw, gh) working size actually used
      scale: (sx, sy) full/work scale factors
      v:     vertical line dicts {x0,x1,y0,y1,center,score}
      h:     horizontal line dicts {y0,y1,x0,x1,center,score}
    """
    H, W = gray.shape
    gw, gh = work_size
    scale = (W / gw, H / gh)
    g = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
    dark = dark_mask(g)
    du8 = (dark * 255).astype(np.uint8)

    # single long-run opening: character-stroke columns score <=0.36 at
    # this scale, ruling lines 0.63-1.0 (VERIFIED both real scans).
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, KERNELS[0]))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNELS[0], 1))
    ver = cv2.morphologyEx(du8, cv2.MORPH_OPEN, kv) > 0
    hor = cv2.morphologyEx(du8, cv2.MORPH_OPEN, kh) > 0
    vb = _line_bands(ver.mean(axis=0), VETO_W)
    hb = _line_bands(hor.mean(axis=1), VETO_W)

    lines_v, lines_h = [], []
    for bd in vb:
        a, b = max(0, bd["a"] - 2), min(gw, bd["b"] + 2)
        colprof = ver[:, a:b].any(axis=1)
        ys = np.flatnonzero(colprof)
        if len(ys) == 0:
            continue
        lines_v.append({"x0": bd["a"], "x1": bd["b"], "center": bd["center"],
                        "score": round(bd["score"], 3),
                        "y0": int(ys.min()), "y1": int(ys.max()) + 1})
    for bd in hb:
        a, b = max(0, bd["a"] - 2), min(gh, bd["b"] + 2)
        rowprof = hor[a:b, :].any(axis=0)
        xs = np.flatnonzero(rowprof)
        if len(xs) == 0:
            continue
        lines_h.append({"y0": bd["a"], "y1": bd["b"], "center": bd["center"],
                        "score": round(bd["score"], 3),
                        "x0": int(xs.min()), "x1": int(xs.max()) + 1})
    return {"work": (gw, gh), "scale": scale, "v": lines_v, "h": lines_h}

