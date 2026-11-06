"""E-N3 (H3): GT-free in-band damage audit on synth arms.

Audit rule (no ground truth needed): a connected component of the
result-vs-input diff inside corridors is FLAGGED if it contains original
ink darker than 150 or is >70 px wide. Ground truth is used only to SCORE
the checker: ground-truth damage = char-mask pixels changed by >40.

Reports per arm: components flagged / components covering true damage /
components over benign paper-only edits (precision & recall).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src import reconstruction, cleanup
import geometry5   # noqa: E402
from eval_synth import load_set          # noqa: E402
from src.reconstruction import corridors          # noqa: E402


def audit(orig, result, lines):
    """Return list of flagged components (x,y,w,h,area,dark)."""
    h, w = orig.shape
    permitted = np.zeros((h, w), np.uint8)
    for _, (x0, x1, y0, y1) in corridors(lines, (h, w)):
        permitted[y0:y1, x0:x1] = 1
    diff = (orig != result) & (permitted > 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        diff.astype(np.uint8), 8)
    flags = []
    for i in range(1, n):
        x, y, ww, hh, area = map(int, stats[i])
        if area < 20:
            continue
        sub = orig[y:y + hh, x:x + ww]
        dark = int((sub < 150).sum())
        if dark > 0 or ww > 70:
            flags.append((x, y, ww, hh, area, dark))
    return flags


def main():
    items = load_set(ROOT / "data/native_synth/dev")
    if not items:
        raise ValueError("No synthetic samples found")
    total = {}
    for arm in ("v3+v4", "v34+g5"):
        total[arm] = dict(tp=0, fn=0, fp=0, benign=0)
    for stem, meta, photo, lm, cm, lines in items:
        v3, *_ = reconstruction.remove_rulings(photo, lines)
        v4, _ = cleanup.cleanup_residuals(photo, v3, lines)
        g5, _r = geometry5.cleanup_residuals(photo, v4, lines)
        results = {"v3+v4": v4, "v34+g5": g5}
        for arm, res in results.items():
            flags = audit(photo, res, lines)
            # true damage sites: components of changed char pixels >40
            dmg_mask = (np.abs(res.astype(np.int16)
                               - photo.astype(np.int16)) > 40) & cm
            n, lab, stats, _ = cv2.connectedComponentsWithStats(
                dmg_mask.astype(np.uint8), 8)
            true_sites = [tuple(map(int, s)) for s in stats[1:]
                          if s[4] >= 20]
            flagged_area = np.zeros(photo.shape, bool)
            for x, y, ww, hh, area, dark in flags:
                flagged_area[y:y + hh, x:x + ww] = True
            tp = sum(1 for (x, y, ww, hh, area) in true_sites
                     if flagged_area[y:y + hh, x:x + ww].any())
            fn = len(true_sites) - tp
            # FP: flags whose bbox has no true-damage pixel at all
            dmg_any = dmg_mask
            fp = sum(1 for f in flags
                     if not dmg_any[f[1]:f[1] + f[3],
                                    f[0]:f[0] + f[2]].any())
            total[arm]["tp"] += tp
            total[arm]["fn"] += fn
            total[arm]["fp"] += fp
            total[arm]["benign"] += int(dmg_mask.sum())
            print(f"{stem} {arm}: true_sites={len(true_sites)} "
                  f"flags={len(flags)} tp={tp} fn={fn} fp={fp}")
    for arm, d in total.items():
        rec = d["tp"] / max(1, d["tp"] + d["fn"])
        prec = d["tp"] / max(1, d["tp"] + d["fp"])
        print(f"{arm}: recall(true damage components flagged)={rec:.3f} "
              f"precision={prec:.3f} (TP={d['tp']} FN={d['fn']} "
              f"FP={d['fp']})")


if __name__ == "__main__":
    main()
