"""geometry5 (E-N2 candidate): junction-aware corridor residue cleanup.

Runs AFTER v3 (+ optionally v4) on its output. Decisions are per connected
component, computed in NATIVE-RESOLUTION strips that extend 40 px beyond the
corridor on both cross-sides, so connectivity of residue to ink OUTSIDE the
corridor is visible (junction-local protection instead of v3's row-wide
flank protection).

Removal gates for a candidate component (ALL must pass):
  G1 area >= 12 px and it lies fully inside the permitted corridor.
  G2 thinness: max distance-transform half-width inside the component
     <= MAX_HALF_W (default 33) -> no stroke-scale bodies.
  G3 cross-width <= CROSS_MAX (70).
  G4 no through-connectivity: the component must not touch the strip
     edges and must not have deep-ink pixels (deficit > 55) in the outer
     context band connected with it (i.e. it is isolated from ink crossing
     the corridor, and from ink reaching past the strip).
  G5 track support on the ORIGINAL image: >= TRACK_T fraction of dark
     (<180) pixels at the component's cross-center over an extended
     along-context (ruling lines continue; strokes terminate).
  G6 flank paper: min flank-paper estimate over the component range
     > FLANK_MIN (180).

Survivors are replaced by the linearly interpolated flank-paper surface
(v4 recipe), only where deficit > 1.5, with 2 px halo dilation. Everything
ambiguous is preserved byte-for-byte.

Knobs are kwargs for ablation. T_INK default 8 (vs v4's 25) probes the
low-contrast blind spots (H2); junction-locality probes H1.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.reconstruction import corridors

CTX = 40          # context px beyond corridor (cross axis)


def cleanup_residuals(original, baseline, lines, T_INK=8.0, MAX_HALF_W=33.0,
                      CROSS_MAX=70, TRACK_T=0.45, FLANK_MIN=180.0,
                      enabled=True):
    """Return (result, records). `baseline` = v3/v4 output; `original` =
    the input scan (track evidence only)."""
    out = baseline.copy()
    records = []
    if not enabled:
        return out, records
    h, w = baseline.shape
    permitted = np.zeros((h, w), np.uint8)
    for _, (x0, x1, y0, y1) in corridors(lines, (h, w)):
        permitted[y0:y1, x0:x1] = 1
    s = min(lines["scale"])

    def rect_to_strip(orient, rect):
        """Expand corridor rect by CTX cross-ward; return strip slices in
        (along, cross) work-space plus the interior column mask."""
        x0, x1, y0, y1 = rect
        if orient == "v":
            X0, X1 = max(0, x0 - CTX), min(w, x1 + CTX)
            return y0, y1, X0, X1, CTX
        Y0, Y1 = max(0, y0 - CTX), min(h, y1 + CTX)
        return x0, x1, Y0, Y1, CTX

    for orient, rect in corridors(lines, (h, w)):
        a0, a1, c0, c1, pad = rect_to_strip(orient, rect)
        if a1 - a0 < 8 or c1 - c0 < 2 * pad + 4:
            continue
        if orient == "v":
            sub_b = baseline[a0:a1, c0:c1]        # rows=along, cols=cross
            sub_o = original[a0:a1, c0:c1]
            interior = np.zeros(sub_b.shape[1], bool)
            interior[pad:sub_b.shape[1] - pad] = True
        else:
            sub_b = baseline[c0:c1, a0:a1].T      # -> (along, cross)
            sub_o = original[c0:c1, a0:a1].T
            interior = np.zeros(sub_b.shape[1], bool)
            interior[pad:sub_b.shape[1] - pad] = True
        # flanks for paper estimation: outermost 4 cols of the strip
        edge = max(2, int(round(3 * s)))
        left = np.percentile(sub_b[:, :edge], 80, axis=1).astype(np.float32)
        right = np.percentile(sub_b[:, -edge:], 80, axis=1).astype(np.float32)
        paper = np.maximum(left, right)
        deficit = paper[:, None] - sub_b
        ink = deficit > T_INK
        # interior-only candidates: restrict labelling region columns
        ink &= interior[None, :]
        if not ink.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), 8)
        dt = cv2.distanceTransform((sub_b < 200).astype(np.uint8) * 255,
                                   cv2.DIST_L2, 3)
        t = np.linspace(0, 1, sub_b.shape[1], dtype=np.float32)
        surf = left[:, None] * (1 - t) + right[:, None] * t
        mask_out = np.zeros(ink.shape, bool)
        # connectivity uses ABSOLUTE darkness on the original image so a
        # depressed paper surface (show-through/dark blotches) cannot mask
        # real ink connections: strokes and punctuation sit far below 150,
        # paper (even darkened by ~45) stays above ~180.
        deep = sub_o < 150
        # components of ORIGINAL deep ink, for the junction-neighbourhood
        # test (glyphs are stroke CLUSTERS; a line corridor region away
        # from glyphs has at most sparse single neighbours).
        _, lab_o, stats_o, _ = cv2.connectedComponentsWithStats(
            deep.astype(np.uint8), 8)
        n_o = stats_o.shape[0]
        nrows = sub_b.shape[0]
        for i in range(1, n):
            st = stats[i]
            # axis0 = along (rows), axis1 = cross (cols); stats = x,y,w,h,area
            r0, c_, hh, ww, area = int(st[1]), int(st[0]), int(st[3]), \
                int(st[2]), int(st[4])
            if area < 12:
                continue
            in_corridor = bool(interior[c_:c_ + ww].all())
            if not in_corridor:
                records.append(_rec(orient, a0, c0, r0, c_, hh, ww, area,
                                    None, None, "preserve", "crosses_edge"))
                continue
            comp = lab[r0:r0 + hh, c_:c_ + ww] == i
            if c_ == 0 or c_ + ww == sub_b.shape[1] \
                    or r0 == 0 or r0 + hh == sub_b.shape[0]:
                records.append(_rec(orient, a0, c0, r0, c_, hh, ww, area,
                                    None, None, "preserve", "touches_strip"))
                continue
            peak = float(dt[r0:r0 + hh, c_:c_ + ww][comp].max())
            # G4b: deep ink in outer context band connected to component
            ctxband = np.zeros(ink.shape[1], bool)
            ctxband[:pad] = True
            ctxband[-pad:] = True
            # dilate component 3 px, check deep-ink touch
            cd = cv2.dilate(comp.astype(np.uint8),
                            np.ones((7, 7), np.uint8)) > 0
            zone_deep = np.zeros(ink.shape, bool)
            zone_deep[r0:r0 + hh, c_:c_ + ww] = cd
            zone_deep &= deep
            connected_out = bool((zone_deep & ctxband[None, :]).any())
            # G5 track on original at component's cross-center
            cc0 = max(0, c_ - 2)
            cc1 = min(sub_b.shape[1], c_ + ww + 2)
            rr0 = max(0, r0 - 48)
            rr1 = min(sub_b.shape[0], r0 + hh + 48)
            track = float((sub_o[rr0:rr1, cc0:cc1] < 180).mean())
            # junction neighbourhood: distinct deep-ink components whose
            # bbox lies within a 25-px vertical buffer of the component
            # and within 30 px horizontally (excluding parts of itself).
            lo = max(0, r0 - 25)
            hi = min(nrows, r0 + hh + 25)
            cx_lo, cx_hi = c_ - 30, c_ + ww + 30
            nb = set()
            for k in range(1, n_o):
                xk, yk, wk, hkk, _ = stats_o[k]
                if yk < hi and yk + hkk > lo and xk < cx_hi and \
                        xk + wk > cx_lo:
                    if not (yk < r0 + hh and yk + hkk > r0 and
                            xk < c_ + ww and xk + wk > c_):
                        nb.add(k)
            many_neighbours = len(nb) >= 3
            flank_ok = (float(left[r0:r0 + hh].min()) > FLANK_MIN
                        and float(right[r0:r0 + hh].min()) > FLANK_MIN)
            elong = (hh >= max(30, 3 * ww))
            safe = (elong and not many_neighbours
                    and peak <= MAX_HALF_W and ww <= CROSS_MAX
                    and not connected_out and track >= TRACK_T
                    and flank_ok)
            records.append(_rec(orient, a0, c0, r0, c_, hh, ww, area,
                                round(peak, 1), round(track, 3),
                                "remove" if safe else "preserve",
                                "gates"))
            if safe:
                m = cv2.dilate(comp.astype(np.uint8),
                               np.ones((5, 5), np.uint8)) > 0
                m &= (deficit[r0:r0 + hh, c_:c_ + ww] > 1.5)
                zone = np.zeros(ink.shape, bool)
                zone[r0:r0 + hh, c_:c_ + ww] = m
                mask_out |= zone
        if mask_out.any():
            fill = np.rint(surf).clip(0, 255).astype(np.uint8)
            if orient == "v":
                tgt = out[a0:a1, c0:c1]
            else:
                tgt = out[c0:c1, a0:a1].T
            tgt[mask_out] = fill[mask_out]
    return out, records


def _rec(orient, a0, c0, r0, c_, hh, ww, area, peak, track, action, why):
    if orient == "v":
        box = [int(c0 + c_), int(c0 + c_ + ww), int(a0 + r0),
               int(a0 + r0 + hh)]
    else:
        box = [int(a0 + r0), int(a0 + r0 + hh), int(c0 + c_),
               int(c0 + c_ + ww)]
    return {"kind": "g5_component", "action": action, "why": why,
            "box": box, "area": int(area), "dt_peak": peak,
            "track": track}