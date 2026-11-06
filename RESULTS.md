# Results and evidence limits

Final audit: 6 November 2026. Results below distinguish direct final reruns,
recorded historical measurements and claims that could not be established.
Private input pixels are not included in this repository.

## Final restoration on the two real pages

The final `restore.py` run regenerated both saved v4 PNG files byte for byte.
Both inputs are 12,700 × 19,367 grayscale pixels, with nine detected vertical
boundaries and two horizontal borders. The fixed continuity limit remains 0.10.
The command completed with **exit 1** because page 2 still fails validation.
Evidence: `evidence/native/final_run.json`; earlier reports are alongside it.

| Direct final measurement | Page 1 | Page 2 |
|---|---:|---:|
| Automated verdict, limited scope | PASS | FAIL |
| Maximum gap-tolerant continuity | 0.0647 | 0.1013 |
| Native median-deficit sum, v3 → v4 | 8 → 8 | 9 → 9 |
| Additional pixels changed by v4 | 128,248 | 88,393 |
| Additional dark pixels changed (<150) | 85,824 | 62,260 |
| Additional changes outside original corridors | 0 | 0 |
| Changes within retained component boxes, v3 → v4 | 0 | 0 |
| Isolated candidates, before → after | 11 → 0 | 7 → 0 |
| Total time, seconds/page | 4.65 | 4.73 |
| Peak process RSS, decimal MB | 1926.6 | 1940.1 |
| Original input hash unchanged | yes | yes |
| Output PNG hash matches saved v4 | yes | yes |

Time includes detection, v3, verification, v4, inventory and output writing.
RSS is the process lifetime maximum on macOS, not a per-stage allocation or a
hardware memory limit. This was not a 16 GB device test. Earlier v4-only times
(2.68/2.64 seconds) excluded generating v3 and should not be compared as if they
measured the same work. Fresh-environment timings can differ.

The native deficit is the difference between a pixel and nearby estimated paper
brightness, on a 0–255 grayscale. Median deficit summarises a strip's cross-axis
profile; its sum is neither a count nor a percentage of removed line pixels.
Occupancy counts the fraction with deficit above 12. Continuity measures the
longest run after bridging short gaps on the reduced 1342 × 2048 image. The
verifier also checks grey profiles at native cross-axis resolution.

The final run checks exact v3 invariance outside permitted corridors and on its
declared protected pixels. The v4 checks compare to v3 and cover retained boxes
and permitted corridors. Neither establishes ownership of removed ink. Zero
remaining actionable candidates means the classifier declines further removal,
not that every ruling pixel is gone. A PASS is not a character-fidelity guarantee.

## Earlier real-page comparison

The retained v3 report compares its preview residual sum with v2:
251.8 → 2.2 on page 1 and 158.2 → 5.0 on page 2 (99.1% and 96.8% reductions
of that contrast proxy, rounded). These are historical measurements, not removal
percentages, and use preview profiles rather than the native sums 8 and 9 above.
The old v2 claim of 96.6–100% removal was measured inside its own mask; later grey
outline failures invalidated an interpretation of complete restoration.

## Overnight native-scale artificial comparison

Twelve development windows, seeds 1000–1011, 2000 × 2800 pixels, every third
window “hard”. The generator uses Songti glyphs, line widths 28–60 pixels,
noise and sometimes JPEG compression. Hard windows add dark paper, stroke-like
arms along boundaries and punctuation on lines. The evaluator supplies known
boundary positions at scale 9.46; it does not test the detector.

The final audit reran all four frozen arms on the saved windows. No parameters
were tuned. These means reproduce the overnight table:

| Arm (12 windows each) | Line proxy | Input-relative preservation | Strong-change pixels/window |
|---|---:|---:|---:|
| Identity control | 0.1248 | 1.0000 | 0 |
| v3 + v4, retained | 0.4008 | 0.9945 | 5215 |
| v3 + v4 + v5, rejected | 0.4032 | 0.9943 | 5288 |
| v3 + v5 | 0.3739 | 0.9943 | 5288 |

Evidence: `evidence/native_synthetic.txt` includes individual windows and times.
The mean pixel counts are truncated to integers by the historical evaluator.

Definitions follow the implementation rather than its earlier inaccurate prose:

- **Line proxy:** among line-mask pixels outside a 15-pixel expansion of character
  masks, the fraction satisfying `240 - result <= 0.2 * (240 - input)`.
  It uses a fixed paper proxy of 240, not a clean image. Very bright pixels can
  satisfy it even without editing, explaining the nonzero identity result.
- **Preservation:** fraction of pixels within a 5-pixel expansion of character
  masks that change by at most 12 grayscale levels from the input.
- **Strong changes:** pixels in that region changing by more than 40 levels.
  The expanded region includes nearby paper; these are damage indicators rather
  than independently adjudicated glyph-loss counts.

One hard window produced 48,904 strong-change pixels with v3+v4. The synthetic
set therefore exposes failures in the retained method too. It has no held-out
native-scale split and underrepresents real dark-flank chains. Its 0.9945 score
cannot be interpreted as 99.45% real character accuracy.

The saved overnight real-image v5 probe reported zero changes on both pages
(508/1284 candidate components retained). That particular real-image v5 probe
was not rerun in the final audit. It is a historical observation; the final
restoration and synthetic comparisons were rerun directly.

## Damage-audit and inventory qualifications

`evidence/damage_audit.txt` records the final rerun of the original audit method.
It flags changed-component boxes containing original ink below 150 or wider
than 70 pixels, then uses box overlap with strong-change character regions.
Historical reported recall was 1.000 (10 v3+v4 sites; 11 v5 sites) and its “precision” score
was about 0.033–0.036. The count excludes sites smaller than 20 pixels, uses box
rather than exact component overlap, and mixes damage-site counts with flag
counts in its precision formula. These figures do not establish a general
perfect-recall detector or calibrated precision. It remains a triage heuristic.

The inventory CSV contains 196 clusters. Its classification CSV gives 100
protected-box overlaps, 84 unchanged/unclassified boxes and 12 edited-leftover
boxes. Whole-box membership can overstate coverage. The inventory frame labels
have a coordinate-scale defect described in RESEARCH.md. None of these categories
is a ground-truth ruling annotation.

## Early 640-pixel artificial baselines

These experiments used a different generator, editing method and clean-reference
metric. For each character-mask pixel, preservation meant result within 12 levels
of clean reference; damage meant a difference above 40. Because inputs themselves
contained noise and overlaid lines, even the identity control scored below 1.0.
Always compare to its control rather than treating damage as editing alone.

The complete `evidence/historical/e9_eval2.csv` has **24 images per arm**:

| Arm | Clean-reference line score | Character score | Damage pixels/image |
|---|---:|---:|---:|
| Control | 0.0092 | 0.7051 | 3611.6 |
| CAND_SAFE | 0.4363 | 0.7008 | 3812.8 |
| CAND_FULL | 0.5288 | 0.6953 | 4071.4 |

CAND_SAFE traded less removal for less additional measured damage. This selected
the older artificial-image pipeline, not the final real-page algorithm.
`e9_eval.csv` has 24 control and only 12 SAFE rows, with no FULL arm.
`e9_eval_2.csv` has four rows per arm. They do not support the previous claim of
two complete 24-image confirmations. The early CSVs are preserved as observations;
the superseded early code and data are retained privately rather than shipped as
another restoration interface. Those early runs are not end-to-end reproducible
from this compact GitHub snapshot alone.

`e4_systems_dev.csv` records 12 images per arm: the early deterministic default
had 5928.4 mean damage pixels, SmolVLM2 the same, LFM the same, and Qwen 5616.8.
That local Qwen improvement did not establish robust tuning value. In
`e5_repair.csv` (two images per arm), all four model arms matched the bad-start
control's 5790 damage pixels; random proposals worsened it to 6145.
The final audit did not rerun any model inference.

Historical E8/E10/E11 verifier notes report constant or poorly discriminating
answers on twelve-pair probes, including 6/12 for both zoomed models. These are
historical narrative results; complete machine-readable prediction tables for
all probes were not available in the retained CSVs. They justify excluding models
from this runtime, not claims about every model or task.

## Final engineering validation

The original 14 deterministic regression tests passed before and after the
reorganisation. Five command-boundary tests cover existing-output refusal,
duplicate stems, missing input, validation-failure reporting and failed image
writes. Final suite: **19 tests, exit 0**, using Python 3.12 on macOS ARM64.
A fresh virtual environment installed the pinned NumPy/OpenCV/Pillow wheels from
PyPI, passed all 19 tests and `pip check`, and reproduced both saved v4 PNGs.
The historical generator also recreated all 48 artificial image/mask/metadata
files byte for byte with the recorded font and seeds. Fresh-clone checks and final preservation receipts are recorded by the owner
outside the repository so the archive need not modify itself after archival.

The source retains the frozen reconstruction, cleanup and verification maths,
with import changes only. Tests are synthetic regressions, not expert review of
all real glyphs. Cross-platform support beyond macOS is untested; the runtime
avoids Apple-only model dependencies and reports memory as unknown when the
platform lacks `resource`.
