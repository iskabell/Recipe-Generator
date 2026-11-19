# My proposed contribution and project plan

Vaidik Nadheria | ACM AI Team 5 | Prep: November 19, 2026

This is Vaidik's work and a proposal to discuss with the team, not an assignment
of teammates or a claim that the whole app is finished.

## Current contribution

My Grocery Store dataset contribution is merged into main (PR #2). I prepared
5,421 RGB images at 224x224, checked their labels and integrity, scanned duplicates,
kept source splits, and supplied manifests, labels, reports and reproducible code.
There are 43 broad classes and 81 detailed classes. No corrupt images or exact
pixel duplicates required removal. A conservative dHash scan found no cross-split
pairs within Hamming distance 2; this does not rule out shared scenes.

## Model research

- CNNs build features from local patterns. ResNet18, MobileNet and the selected
  YOLO classification model are CNN approaches, not alternatives to the CNN family.
- Image transformers (e.g. DeiT) compare image patches using attention. A pretrained
  DeiT-Tiny is a reasonable future comparison, but the assigned comparison is
  ResNet18 vs YOLO26, so it is the priority for this standup.
- ResNet18: pretrained visual features plus a new classifier for our 43 classes.
- YOLO26n-cls: the classification version of YOLO26; predicts one class for the
  whole photo. A YOLO26 detector is a different task and requires bounding-box
  annotations we do not have in the Grocery Store dataset.
- MobileNetV3-Small: a future lightweight option if laptop latency is a problem.

## My experiment

`compare_models.py` uses independently initialized linear heads over two frozen
pretrained feature extractors: ResNet18 and YOLO26n-cls. It extracts features once
and trains only the new head. This is an initial transfer-learning comparison,
not full-network fine-tuning or a claim about which architecture is always best.
Both models see the same images and the same 43-class vocabulary. The input is
our full, letterboxed 224x224 image. ResNet uses ImageNet normalization; YOLO uses
RGB values scaled to [0,1], matching its classification preprocessing convention.
No augmentation is used in this frozen-feature baseline. Future fine-tuning may
add it, on training images only.

The 2,640 original training images are divided into 2,265 training and 375
stratified development images with seed 42. The original 2,485 test images remain
outside training and epoch selection. The original 296-image validation set is
not used because it lacks six categories. Every new development category is
represented. Split manifests and hashes are saved so the team can reuse them.
Related product/scene images can still cross the random development split, so
those development results should not be treated as independent real-world proof.

Epoch selection uses development macro F1. Final metrics include top-1 accuracy,
top-5 accuracy, macro F1, per-class recall and batch-1 CPU forward latency. The
latency excludes image loading, preprocessing and UI, and is specific to the
recorded machine. Keep a test set untouched for future final team selection;
our standup comparison is preliminary and repeated tuning on this test would
weaken its value as an unbiased final evaluation.

## Proposed application flow

Photo or typed list -> ingredient identification -> user confirms inventory ->
normalize ingredient names -> rank recipes -> show matches and missing items.

Start with individual ingredient photos and text input. The current labels do
not support identifying every object in one fridge image. Also, broad labels
such as potato, juice and yogurt hide varieties/flavors. Preserve fine labels
and ask users for details before matching recipes. Keep dairy and plant milks
separate. Quantities cannot be inferred reliably from these class labels.

My suggested next ownership is the inventory-normalization and recipe-ranking
backend, if the team agrees. Build a simple ingredient-overlap baseline first.
Show missing ingredients, don't assume pantry staples, and support explicit
dietary preferences. The merged AllRecipes CSV includes ingredients and summary,
but no directions or recipe URL columns. A complete cooking demo needs a source
with instructions; do not invent missing steps.

## Proposed next milestones (confirm at prep)

1. Nov 19-20: present individual data/model progress and agree on interfaces,
   vocabulary, shared splits and owners.
2. Nov 21-27: integrate confirmed ingredient inventory with a small recipe index;
   collect team phone photos for a separate real-world test.
3. Nov 28-Dec 4: compare complete user flow, audit label mismatches and false
   predictions, handle uncertain and unknown items.
4. By the early-Winter showcase: polish a reproducible demo, record actual metrics,
   acknowledge limits and prepare a short backup recording or offline walkthrough.

## Decisions I should raise at the prep meeting

- Who is owning the YOLO branch and who is owning ResNet? Share this initial
  comparison rather than overwriting or claiming another person's model work.
- Are we doing classification of individual items now, with detection later?
- Can we use a shared development split and identical ingredient vocabulary?
- Which recipe data source includes complete instructions and source links?
- Can I own normalization/ranking next, and who will connect that with the UI?
- Which slide should I speak on, and how long do I have? Proposed: 45-60 seconds.

## Sources

- Current repository: https://github.com/iskabell/Recipe-Generator
- My data contribution: Data/grocery_store/README.md and processed/summary.json
- PyTorch transfer learning: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- YOLO classification: https://docs.ultralytics.com/tasks/classify/
- YOLO26 model tasks: https://docs.ultralytics.com/models/yolo26/
- DeiT: https://huggingface.co/docs/transformers/model_doc/deit

No teammate results are presented as my own. See results/metrics.json for measured
results from this experiment; if a model fails, report that failure instead.
