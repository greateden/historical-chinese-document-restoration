"""Restore the supported eight-column grayscale sheets with the frozen v3+v4 method."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from src.detection import detect_rulings
from src.reconstruction import corridors, remove_rulings
from src.cleanup import cleanup_residuals
from src.verification import native_residual_metrics, residual_metrics, verify_removal

WORK = (1342, 2048)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def peak_rss_mb():
    """Process lifetime peak; resource uses bytes on macOS, KiB on Linux."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e6 if sys.platform == 'darwin' else 1e6 / 1024), 1)


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write image: {path}')


def summary(gray, lines):
    bands, _ = residual_metrics(gray, lines)
    native = native_residual_metrics(gray, lines)
    return dict(bands=bands, native=native,
                median_deficit_sum=round(sum(r['median_deficit'] for r in native), 3),
                max_continuity=max(r['run_fraction'] for r in bands))


def process(path, outdir):
    start = time.perf_counter()
    digest = sha256(path)
    original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f'Cannot read image: {path}')
    lines = detect_rulings(original)
    if (len(lines['v']), len(lines['h'])) != (9, 2):
        raise ValueError('Unsupported page: expected nine vertical and two horizontal boundaries')

    baseline, edit, protect, zones, stats = remove_rulings(original, lines)
    baseline_report, _ = verify_removal(original, baseline, lines, edit, protect, stats)
    # Full-page masks are no longer needed once the v3 checks are recorded.
    del edit, protect
    before = summary(baseline, lines)
    result, records = cleanup_residuals(original, baseline, lines)
    after = summary(result, lines)
    remaining_image, remaining = cleanup_residuals(original, result, lines)
    del remaining_image

    permitted = np.zeros_like(original)
    for _, (x0, x1, y0, y1) in corridors(lines, original.shape):
        permitted[y0:y1, x0:x1] = 1
    changed = dark_changed = outside = 0
    for y in range(0, len(original), 128):
        delta = baseline[y:y+128] != result[y:y+128]
        changed += int(delta.sum())
        dark_changed += int((delta & (baseline[y:y+128] < 150)).sum())
        outside += int((delta & (permitted[y:y+128] == 0)).sum())
    del permitted
    protected_changes = 0
    connectivity = []
    for record in records:
        if record['action'] != 'preserve':
            continue
        x0, x1, y0, y1 = (record[k] for k in ('x0', 'x1', 'y0', 'y1'))
        a, b = baseline[y0:y1, x0:x1], result[y0:y1, x0:x1]
        protected_changes += int(np.count_nonzero(a != b))
        connectivity.append([int(cv2.connectedComponents((v < 150).astype(np.uint8), 8)[0]-1)
                             for v in (a, b)])
    high = lambda entries: sum(r['action'] == 'remove' for r in entries)
    # Final residual checks supersede baseline residual failures. Other v3 safety
    # checks must still hold. Counts are diagnostics, never proof of glyph safety.
    residual_checks = {'native_grey_residual', 'independent_grey_residual'}
    checks = dict(
        baseline_safety=all(c['pass'] for c in baseline_report['checks']
                            if c['check'] not in residual_checks),
        outside_corridors_unchanged=outside == 0,
        retained_components_unchanged=protected_changes == 0,
        retained_connectivity_unchanged=all(a == b for a, b in connectivity),
        native_grey=all(r['pass'] for r in after['native']),
        fixed_residual_thresholds=all(r['pass'] for r in after['bands']),
        grey_no_regression=all(b['median_deficit'] <= a['median_deficit'] and
                               b['grey_occupancy'] <= a['grey_occupancy']
                               for a, b in zip(before['native'], after['native'])))
    stem = path.stem
    write_image(outdir / f'{stem}.png', result)
    previews = [cv2.resize(v, WORK, interpolation=cv2.INTER_AREA) for v in (original, result)]
    write_image(outdir / f'{stem}_comparison.png', cv2.hconcat(previews))
    view = cv2.cvtColor(previews[1], cv2.COLOR_GRAY2BGR)
    sx, sy = lines['scale']
    for r in remaining:
        cv2.rectangle(view, (round(r['x0']/sx)-3, round(r['y0']/sy)-3),
                      (round(r['x1']/sx)+3, round(r['y1']/sy)+3), (0, 170, 255), 2)
    write_image(outdir / f'{stem}_ambiguities.png', view)
    _, heat = residual_metrics(result, lines)
    write_image(outdir / f'{stem}_residual_heatmap.png', cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO))
    checks['input_sha256_unchanged'] = sha256(path) == digest
    return dict(input=path.name, input_sha256=digest, output_sha256=sha256(outdir / f'{stem}.png'),
                shape=list(original.shape), algorithm='v3 reconstruction + v4 residual cleanup',
                geometry=lines, baseline=baseline_report, before=before, after=after,
                n_ambiguous=len(zones), ambiguous_zones=zones,
                component_counts_before=dict(Counter(r['kind'] for r in records)),
                component_counts_after=dict(Counter(r['kind'] for r in remaining)),
                isolated_candidates_before=high(records), isolated_candidates_after=high(remaining),
                changed_pixels_from_v3=changed, dark_pixels_changed_from_v3=dark_changed,
                outside_corridor_changes_from_v3=outside, retained_component_changes=protected_changes,
                retained_connectivity_pairs=connectivity, decisions=records, remaining=remaining,
                checks=checks, passed=all(checks.values()),
                seconds=round(time.perf_counter()-start, 2), peak_rss_mb=peak_rss_mb(),
                glyph_evidence='No clean glyph reference. In-band character preservation is unproven.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--outdir', required=True, type=Path, help='A new output directory')
    args = parser.parse_args(argv)
    try:
        if args.outdir.exists():
            raise FileExistsError('Refusing existing output directory')
        stems = [p.stem.casefold() for p in args.inputs]
        if len(set(stems)) != len(stems):
            raise ValueError('Input filenames must have distinct stems')
        for path in args.inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
            if args.outdir.resolve() in path.resolve().parents:
                raise ValueError('Output directory must not contain an input')
        args.outdir.mkdir(parents=True, exist_ok=False)
        pages = []
        for path in args.inputs:
            report = process(path, args.outdir)
            pages.append(report)
            print(f"{path.name}: passed={report['passed']}, {report['seconds']} s", flush=True)
        report = dict(passed=all(p['passed'] for p in pages), pages=pages)
        (args.outdir / 'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        return 0 if report['passed'] else 1
    except (OSError, ValueError, cv2.error) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
