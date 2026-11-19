# Vaidik's measured model results

These are my new runs, not the results from a teammate's branch.

| Model | Test accuracy | Macro F1 | Median CPU forward time |
| --- | --- | --- | --- |
| ResNet18 | 77.8% | 0.702 | 24.8 ms |
| YOLO26n classification | 81.6% | 0.741 | 7.4 ms |

Same split: 2,265 training / 375 development / 2,485 test images; 43 classes.
The original 296-image validation set was excluded because six classes are absent.
Pretrained backbones were frozen. I trained a new linear head for each, using
80 epochs, AdamW, seed 42; the best epoch was selected using development macro F1.
ResNet best epoch: 36; YOLO best epoch: 21.
No augmentation or full-network fine-tuning was used in this initial comparison.
YOLO performed 3.8 percentage points better on this test, and was
faster on this CPU. This supports trying it next; it is not proof of universal
superiority. Do not reuse the test to repeatedly select future changes.

## Short result sentence for standup

I compared pretrained ResNet18 and YOLO26 classification on the same grocery
images, training new classifier heads. ResNet got 77.8% test accuracy and YOLO
got 81.6%. These are preliminary single-item results. I would next test on
new kitchen photos and connect confirmed ingredients to recipe matching.

## Important limits

The random development split can share related product/scene images with
training. Single-source test images are not independent kitchen/fridge validation.
Both classifiers output one broad category for a whole image. A full-fridge
object detector needs different labels. The latency is batch-1 CPU forward time
only, excluding image preprocessing, downloads and UI. Softmax scores are not
calibrated probabilities of correctness. Results are from one seed.

## Artifacts

`results/metrics.json`, `experiment_splits.csv`, prediction JSON files,
confusion matrices and trained head files document the run. The official pretrained
backbones are downloaded separately when running live inference.
