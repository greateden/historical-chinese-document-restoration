"""E-N1/E-N2 harness: native-scale synthetic evaluation.

Metrics (mask-based, no clean reference; control arm gives the floor):
  removed  = fraction of target line pixels with >=80% deficit reduction
             relative to the fixed paper proxy 240
             target = line_mask & ~dilate(char_mask,15)  [pure-line zone]
  pres     = fraction of dilate(char_mask,5) with |delta| <= 12
  dmg      = count in dilate(char_mask,5) with |delta| > 40
  pres_i   = same preservation inside line_mask∩char zone (intersections)
Arms: control (identity), v3+v4 pipeline, geometry5 variants.
Read-only on data; writes nothing except printed table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


def zones(shape, lm, cm):
    inter = cv2.dilate(cm.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    target = lm & ~inter
    pres_zone = cv2.dilate(cm.astype(np.uint8), np.ones((11, 11),
                                                         np.uint8)) > 0
    return target, pres_zone


def metrics(photo, result, lm, cm):
    target, pres_zone = zones(photo.shape, lm, cm)
    t = photo[target].astype(np.int16)
    r = result[target].astype(np.int16)
    removed = float(((240 - r) <= 0.2 * (240 - t)).mean()) if t.size else 1.0
    pz = photo[pres_zone].astype(np.int16)
    rz = result[pres_zone].astype(np.int16)
    d = np.abs(rz - pz)
    pres = float((d <= 12).mean()) if pz.size else 1.0
    dmg = int((d > 40).sum())
    inter_zone = pres_zone & lm
    if inter_zone.any():
        di = np.abs(result[inter_zone].astype(np.int16)
                    - photo[inter_zone].astype(np.int16))
        pres_i = float((di <= 12).mean())
    else:
        pres_i = 1.0
    return removed, pres, dmg, pres_i


def arms_for(geom5):
    arms = {"control": lambda photo, lines: photo}

    def v34(photo, lines):
        from src import reconstruction
        from src import cleanup
        out, *_ = reconstruction.remove_rulings(photo, lines)
        out, _rec = cleanup.cleanup_residuals(photo, out, lines)
        return out

    arms["v3+v4"] = v34
    if geom5:
        def g5(photo, lines):
            from src import reconstruction
            from src import cleanup
            import geometry5
            out, *_ = reconstruction.remove_rulings(photo, lines)
            out, _rec = cleanup.cleanup_residuals(photo, out, lines)
            out, _rec = geometry5.cleanup_residuals(photo, out, lines)
            return out
        arms["v34+g5"] = g5

        def g5only(photo, lines):
            from src import reconstruction
            import geometry5
            out, *_ = reconstruction.remove_rulings(photo, lines)
            out, _rec = geometry5.cleanup_residuals(photo, out, lines)
            return out
        arms["v3+g5"] = g5only
    return arms


def load_set(set_dir: Path):
    items = []
    S = 9.46  # match the real page's native->work scale so strip windows
    # (edge/block/halo = f(scale)) have the same PHYSICAL size as on the
    # real scans; corridor rects are stored in native px and converted
    # down to work px here.
    for f in sorted(set_dir.glob("*_photo.png")):
        stem = f.name[:-10]
        photo = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        lm = cv2.imread(str(set_dir / f"{stem}_line_mask.png"),
                        cv2.IMREAD_GRAYSCALE) > 0
        cm = cv2.imread(str(set_dir / f"{stem}_char_mask.png"),
                        cv2.IMREAD_GRAYSCALE) > 0
        meta = json.loads((set_dir / f"{stem}_meta.json").read_text())

        def conv(c, orient):
            if orient == "v":
                return dict(x0=int(c["x0"] / S) - 1, x1=int(c["x1"] / S) + 2,
                            y0=int(c["y0"] / S), y1=int(c["y1"] / S) + 1,
                            center=int((c["x0"] + c["x1"]) / 2 / S),
                            score=1.0)
            return dict(x0=int(c["x0"] / S), x1=int(c["x1"] / S) + 1,
                        y0=int(c["y0"] / S) - 1, y1=int(c["y1"] / S) + 2,
                        center=int((c["y0"] + c["y1"]) / 2 / S), score=1.0)

        lines = {"scale": (S, S), "work": None,
                 "v": [conv(c, "v") for c in meta["corridors"]
                       if c["orient"] == "v"],
                 "h": [conv(c, "h") for c in meta["corridors"]
                       if c["orient"] == "h"]}
        items.append((stem, meta, photo, lm, cm, lines))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--g5", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    items = load_set(ROOT / args.set)
    if not items:
        raise ValueError("No synthetic samples found")
    arms = arms_for(args.g5)
    tot = {a: dict(rem=[], pres=[], dmg=[], pi=[]) for a in arms}
    for stem, meta, photo, lm, cm, lines in items:
        for name, fn in arms.items():
            t0 = time.time()
            try:
                res = fn(photo.copy(), lines)
            except Exception as e:
                print(f"{stem} {name}: ERROR {type(e).__name__} {e}")
                raise
            rem, pres, dmg, pi = metrics(photo, res, lm, cm)
            tot[name]["rem"].append(rem)
            tot[name]["pres"].append(pres)
            tot[name]["dmg"].append(dmg)
            tot[name]["pi"].append(pi)
            tag = f"{stem}{'*' if meta['hard'] else ' '}{name:>8}: "
            print(f"{tag} removed={rem:.3f} pres={pres:.4f} dmg={dmg:6d} "
                  f"pres_i={pi:.4f} {time.time()-t0:.1f}s")
            del res
    print("\n== means ==")
    for name, d in tot.items():
        print(f"{name:>9}: removed={np.mean(d['rem']):.4f} "
              f"pres={np.mean(d['pres']):.4f} "
              f"dmg={int(np.mean(d['dmg']))} "
              f"pres_i={np.mean(d['pi']):.4f}")


if __name__ == "__main__":
    main()
