# Research record

## Questions and scope

The project asked whether printed ruling lines could be reduced without erasing
Chinese character strokes, whether small local AI models helped choose settings
or judge damage, and how to check results when no clean original exists.
The research is complete. This document consolidates the useful findings rather
than preserving the order of development.

There were two kinds of evidence. Early artificial 640 × 640 images had a clean
reference and masks marking the intended lines and strokes. Later tests used two
private 12,700 × 19,367 scans without a clean reference. An overnight artificial
set of twelve 2000 × 2800 windows was designed to match the larger stroke scale.
These sets answer different questions and their percentages are not comparable.

## Approaches investigated

The early pipeline estimated the paper brightness, selected dark pixels, found
long horizontal and vertical structures, protected thick ink cores and filled
selected pixels using OpenCV's Telea method. Telea estimates missing values from
nearby image content. Longer detection kernels and narrower removal masks traded
line removal against unwanted stroke changes. Wider masks and flat filling did
not solve that trade-off.

Small vision-language models proposed settings through a restricted interface.
Experiments included SmolVLM2-2.2B, Qwen3-VL-2B and 4B quantisations, and
LFM2.5-VL-1.6B. They did not demonstrate dependable additional value in this setup.
A Qwen tuning arm improved some development examples relative to an early default,
but deliberate bad-start tests did not show successful repairs. Structured replies
were easier to obtain than reliable visual judgement. Model verification tests
used very small samples and configuration labels, rather than expert judgements
of every character; they cannot support a claim that all AI verification fails.
The models and their dependencies are therefore absent from the final runtime.

The first real scans exposed a scale error. Small background windows fitted
inside thick grid lines, treating them as background. Thick-line protection also
mistook line cores for character ink. The old checker examined only the proposed
removal mask and could approve a near-empty edit. Synthetic success had not
established real-page success.

The v2 method separated detection on a smaller image from editing at full size.
It substantially changed the pages, but its dark-pixel criteria missed grey edges
and hollow outlines. Its historical removal percentages are restricted to its
own mask and are not an independent measure of complete line removal.

The v3 method retained that detector and estimated paper from both sides of each
line strip. It followed local grey profiles, included faint edges, and protected
rows with foreground entering from either side. It restored all declared
protected pixels after processing crossing strips. The independent checker
examined expected geometry rather than relying on the edit mask alone.

The v4 method added a conservative pass over v3 results. Connected groups of dark
pixels were classified using position, shape, isolation, support in the original
scan and paper brightness. Some isolated dots near the frame and long vertical
fragments were replaced. Uncertain components stayed unchanged.

## What the overnight work added

The overnight inventory counted 196 residual clusters under a specific geometric
and brightness rule. A later classification found 100 boxes overlapping protected
regions, 84 classified as unchanged/unclassified and 12 as edited with residue.
These are bounding-box categories, not a precise labelling of every residual
pixel. They must not be read as 196 proven removable line fragments.
The frame-dot tagging in the inventory probe mixed working and native coordinate
units; its frame-specific labels should not be relied on. The v4 runtime uses a
separate native-coordinate classifier.

Measured stroke and line widths overlapped. Thickness alone could not safely
separate them. The twelve new artificial windows included dark paper, long
stroke-like structures beside lines and punctuation on lines. These ingredients
exposed harmful edits that simpler examples missed. The set still underrepresented
some real connected residue classes and had no separate held-out evaluation.

The v5 candidate replaced broad protection with more local shape and connection
rules and considered fainter residue. Revisions reduced some observed damage,
but the final comparison still showed a small increase in strong changes near
synthetic characters. It changed zero pixels on the two real v4 outputs in the
saved overnight probe. It was rejected. The final audit repeated the saved
synthetic comparison without tuning anything.

A proposed automatic damage audit flagged edits containing dark original pixels
or wide changed regions. It found the tested damage sites but raised many alarms
on benign edits. Its implementation compares bounding boxes and counts different
units for true positives and false positives. Its reported “precision” is a
heuristic score, not a standard detection precision estimate. It is evidence for
manual triage only, not a dependable automatic safety gate.

## Why retain v3 followed by v4?

This is the only final combination with native results, explicit residual checks,
recorded component decisions, preservation regressions and a reproducible match
to the saved v4 images on both private pages. It removes some additional residue
without changing the existing continuity thresholds. It remains a partial result;
page 2 fails. “Best validated” means best supported for these pages among the
available implementations, not proven optimal or universally safe.

The final audit changed the command and file organisation, not the restoration
formulae or thresholds. It checked that the renamed reconstruction, cleanup and
verification modules matched their originals apart from imports. It removed the
obsolete v2 editor while retaining its detector. The single command now computes
its own v3 baseline, checks image-write errors, refuses existing directories and
colliding input names, and handles process-memory reporting by platform.
The historical synthetic harness was corrected so omitting `--g5` really excludes
v5; empty sets and evaluation failures no longer silently produce a success table.

## Evidence retained and claims corrected

Numeric CSVs retain the early comparisons, including incomplete runs. Native
JSON reports retain measurements, checks, image hashes and decisions, without
scan pixels. Four small method files retain the overnight generator, evaluator,
rejected candidate and damage-audit definition. Their runtime is separate from
`restore.py`. Original development logs, old source, private scans, generated
images and old Git objects were preserved in a verified local backup.

The final documentation drops several unsupported statements: complete grid
removal; perfect character preservation; a 16 GB deployment guarantee inferred
from process memory; and two complete held-out confirmations. In particular,
`e9_eval.csv` has unequal arm counts and `e9_eval_2.csv` has only four samples per
arm. Only `e9_eval2.csv` contains 24 samples for every reported arm.

No outside literature is required to substantiate these local findings. Earlier
notes contained remembered or snippet-level references; they are not promoted
into verified citations here.

## Unresolved questions

There is no expert-adjudicated real-page damage reference, generalisation study,
large held-out native-scale set or transcription evaluation. Foreground wholly
inside a ruling strip is particularly ambiguous. Brightness checks cannot
resolve the ownership of overlapping ink. More representative data and expert
labels would be needed before making stronger claims. They are future research
possibilities, not work undertaken for this archive.
