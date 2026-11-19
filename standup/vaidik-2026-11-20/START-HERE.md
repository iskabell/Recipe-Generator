# Vaidik's standup contribution

This folder contains my work for the November 19 prep meeting and November 20
ACM AI standup. It does not replace or claim a teammate's work.

## Ready to present

- Open `demo.html` in a browser. This is an offline walkthrough of real saved
  predictions, not live inference. Switch models and show a mistake as well.
- Use `MY-SPEAKING-SCRIPT.txt` for the short update.
- `MEETING-PREP.md` contains the meeting details and likely questions.
- `PROJECT-PLAN.md` contains model research, methodology and my proposed next role.
- `RESULTS.md` explains the measured comparison and its limits.
- `Vaidik-Standup-Slides.pdf` is the visually checked presentation backup.
- Personal editable slides:
  https://docs.google.com/presentation/d/1kyP8WwB52lkBGm_UPW1w1z3c2BtBmpPCrO3ntf63sZg/edit

The shared team template is unchanged. My copy also has speaker notes.
The native slides and PDF were visually checked. Keep the PDF as an offline backup.

## Actual results

| Model | Test accuracy | Macro F1 |
| --- | --- | --- |
| ResNet18 | 77.8% | 0.702 |
| YOLO26n classification | 81.6% | 0.741 |

Both runs used frozen pretrained backbones and newly trained linear heads.
This is an initial single-item classification experiment, not multi-item detection.
The same 2,485 test images were used. The scripts do not reuse a teammate's weights.

## Optional: run a new photo locally

Run these commands from the `Recipe-Generator` repository, with Python 3.12:

```bash
python3 -m venv .venv-standup
source .venv-standup/bin/activate
python -m pip install -r standup/vaidik-2026-11-20/requirements.txt
python standup/vaidik-2026-11-20/predict_image.py /path/to/ingredient.jpg --model yolo26n
python standup/vaidik-2026-11-20/predict_image.py /path/to/ingredient.jpg --model resnet18
```

The first model load downloads official pretrained backbones. The trained heads
are included. The offline HTML walkthrough needs no installation or internet.
Exact recorded environment versions are in `results/environment.json`.

## Optional: reproduce the experiment

The repository's processed image archives must be unpacked first:

```bash
python Data/grocery_store/unpack_images.py
python -m unittest discover -s Data/grocery_store/tests/
```

The measured results are already included. To preserve them, first rename the
existing `results` folder to an unused backup name, then run:

```bash
python standup/vaidik-2026-11-20/compare_models.py --models resnet18 yolo26n
```

Default settings: 80 epochs, seed 42, 2,265 training and 375 development images;
best epoch selected by development macro F1. Feature extraction uses batch 32;
classifier training uses batch 256. CPU forward timings exclude preprocessing.

## Verification performed

- All five existing preprocessing tests passed.
- Saved predictions reproduce both reported accuracy and macro F1 values.
- Training, development and test paths are disjoint; all contain 43 classes.
- Both saved heads successfully ran individual-image inference.
- Four PDF pages were rendered and visually inspected.

New GitHub publishing was blocked by the connected account's integration
permissions (403). This work is committed locally on its own branch; nothing
was pushed or merged into the team's repository by this preparation.
