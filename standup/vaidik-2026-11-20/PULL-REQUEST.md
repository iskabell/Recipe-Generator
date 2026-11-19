# Add Vaidik's standup model comparison and prep materials

Adds my independent ResNet18 and YOLO26n classification comparison for the
recipe generator standup, plus a personal project plan, speaking notes, slides
and an offline walkthrough of recorded predictions.

Both models use frozen official pretrained backbones with newly trained linear
heads. On the same 2,485 test images, ResNet18 reached 77.8% accuracy / 0.702
macro F1 and YOLO26n reached 81.6% / 0.741. The 375-image development split comes
only from the original training split. Epoch selection uses development macro F1.
These preliminary single-item results do not establish multi-item detection or
performance on new kitchen photos.

The change is confined to `standup/vaidik-2026-11-20/`. It includes trained heads,
split manifests, saved predictions, confusion matrices and exact run versions.
It does not change team datasets, a teammate's models or the shared slides.

Validation: five existing preprocessing tests pass; saved predictions reproduce
reported metrics; split paths are disjoint; both heads run local inference;
embedded demo images and demo switching/filter logic checked; slides rendered
and visually reviewed. A full browser test could not run in the execution
environment, so the offline demo received script-level checks.
