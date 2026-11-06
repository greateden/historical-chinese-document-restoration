# Historical Chinese document restoration

This project reduces printed grid lines on scanned Chinese character sheets.
It is a finished research artifact, rather than a general document repair tool.
It was developed and checked on two private pages, each 12,700 × 19,367 pixels.
The original scans are deliberately excluded.

## Why this is difficult

Grid lines and character strokes can have the same colour and thickness.
Where they touch, a program cannot always tell which ink belongs to the text.
Removing every visible line can therefore erase part of a character.
This project gives preservation priority and leaves uncertain areas for review.

## How it works

The program uses traditional image processing with NumPy and OpenCV:

1. It makes a smaller copy to find long, narrow grid boundaries. The supported
   layout has nine vertical boundaries and two horizontal borders.
2. It returns to the full-size scan and examines narrow strips around those
   boundaries. Nearby paper on both sides supplies an estimate of the background.
3. It replaces likely line pixels with that estimated paper colour. Where dark
   strokes enter from the sides, it protects the area and records the uncertainty.
4. A conservative second pass removes some isolated dots and vertical fragments.
   Nearby ink, dark paper and unclear connections can block removal.
5. Separate checks look for remaining lines and changes outside allowed areas.

The retained implementation is the **v3 reconstruction followed by v4 cleanup**.
It uses no AI models, OCR, GPU or network at runtime. AI assistants helped develop
and review the project. Earlier experiments did run small vision-language models
(models that process images and text), but their measured results did not justify
including them in the final program.

## What the research found

Working at the right image scale mattered more than adding an agent. An early
method tuned on small artificial images barely changed the full-size scans.
The later method removed much of the visible grid and its grey outlines.
The final audit reproduced the saved v4 PNGs byte for byte on both private pages.

This is **partial restoration**. Page 1 passes the automated checks; page 2 still
fails because a retained fragment exceeds the fixed continuity limit
(0.1013 versus 0.10). Passing checks does not prove every character is intact.
There is no clean copy of either page to establish that.

The overnight experiments found 196 residual clusters under their inventory rule.
A proposed v5 cleaner made no changes on the real pages and slightly increased
changes to character regions in artificial tests, so it was rejected. A proposed
damage checker produced too many false alarms for automatic approval.
[RESEARCH.md](RESEARCH.md) explains the lessons and [RESULTS.md](RESULTS.md) gives
measurements, definitions and evidence limits.

## Run the program

Use Python 3.12 or later. From the repository directory, create an isolated
environment and install the two pinned packages:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python restore.py --outdir outputs/my-run /path/to/page1.jpg /path/to/page2.jpg
```

Replace the example input paths with your own authorised scans. On Windows use
`python` to create the environment and `.venv\Scripts\python.exe` for the other
commands. Windows and Linux execution have not been tested.

These commands create a local environment, install NumPy and the headless
OpenCV wheel from PyPI, run tests, and write results into a new directory.
They do not run a downloaded shell script. Package sources were checked against
[NumPy's installation guide](https://numpy.org/install/),
[NumPy on PyPI](https://pypi.org/project/numpy/), and
[OpenCV's Python wheel documentation on PyPI](https://pypi.org/project/opencv-python-headless/).
Use only one OpenCV wheel variant in an environment.

The output directory must not already exist, and input names must have different
stems (for example, `page1` and `page2`). Inputs are read as 8-bit grayscale.
The program writes restored PNGs, input/result comparisons, amber boxes around
remaining cleanup candidates, residual heatmaps, and `report.json`. The JSON also
lists protected v3 zones; the boxes are not an exhaustive damage annotation.
Input hashes are checked again after processing. Keep originals as archival masters.

Exit code **0** means the limited checks passed, **1** means a completed run
needs review, and **2** means an input or output error. An interrupted or failed
run may leave partial files; use a new directory for the next attempt.

## Reproduce the evaluation

The unit tests need no private data or models. To repeat the two real-page
measurements, the owner must supply the exact private scans; their hashes and
recorded results are in `evidence/native/`. Without them, this part cannot be
independently reproduced from GitHub alone.

The retained overnight comparison can be regenerated separately. It needs Pillow
and the macOS Songti font used in the original experiment:

```sh
.venv/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: -r requirements-evaluation.txt
.venv/bin/python evidence/methods/gen_synth.py --out data/native_synth/dev --n 12 --seed 1000
.venv/bin/python evidence/methods/eval_synth.py --set data/native_synth/dev --g5
.venv/bin/python evidence/methods/check_synth_audit.py
```

The generator creates artificial windows, not copies of the private scans.
It retains the historical font selection; different fonts or library versions
may change results. The evaluator supplies known boundary positions, so this
comparison tests editing rather than detection. The rejected v5 code is kept
only in `evidence/methods/` to make this negative result inspectable.

## Limitations and possible future work

The method assumes straight, mostly regular grids like the two tested sheets.
It can leave faint outlines, connected line fragments and frame dots. It alters
paper texture and may damage character content wholly inside a line strip.
Artificial tests expose some such failures but do not establish real transcription
accuracy. Large scans still require substantial memory; see RESULTS.md for the
measured process peak rather than a universal hardware requirement.

Future researchers could obtain expert labels for character-line intersections,
test more page layouts, and develop more representative evaluation data. Those
questions remain open; no further research is scheduled in this archive.

## Files and rights

```text
restore.py                    one restoration command
src/                          detection, reconstruction, cleanup, verification
tests/                        algorithm and command safety regressions
evidence/native/              private-page measurements, without images
evidence/historical/          original numeric experiment tables
evidence/methods/             overnight comparison and rejected method
README.md, RESEARCH.md, RESULTS.md
requirements.txt              runtime packages
requirements-evaluation.txt   optional historical evaluation package
```

Private scans, generated images, environments, model weights and old Git history
are excluded. The owner retains a separate local preservation backup.
No software licence has been authorised or granted for this project.
Third-party dependencies have their own licences. This archive is not an
assertion of permission to redistribute source scans or fonts.
