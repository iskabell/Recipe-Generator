# Vaidik's standup and prep notes

## Where and when

- Prep: Thursday, November 19, 2026, 4 PM, Geisel basement, room 1040.
- Standup: Friday, November 20, 2026, 6 PM PST, virtual ACM Discord voice channel.
- Confirm the exact Discord voice channel with the team. Mentors support; members
  explain their own contributions. Timing comes from the supplied channel notes.

## My short update for the prep meeting

I finished preparing the grocery dataset, and my work is merged into main.
There are 5,421 images across 43 broad categories. The original validation split
is missing six categories, so the model comparison uses a development split from
training data. I compared frozen ResNet18 and YOLO26 classification features,
training a new classifier head for each. ResNet reached 77.8% test accuracy and
YOLO reached 81.6% on the same 2,485 test images. I also made an offline demo of
the saved predictions.
I'd like to help with the ingredient-matching backend next, if that fits our plan.

## My standup script (data section, about 40 seconds)

I worked on preparing the ingredient image dataset. It contains 5,421 grocery
photos across 43 broad categories. I checked the images and labels, looked for
duplicates, and converted the images to a consistent 224-by-224 RGB format while
keeping their proportions. The scripts, images and reports are merged into our
repository. One issue I found is that six categories are missing from the
original validation split, so our model experiments need a development split
that includes all categories. The next step is connecting reliable ingredient
predictions to recipe matching.

## Demo fallback (no live training needed)

Open demo.html in a browser. It shows real dataset examples, recorded model
predictions and measured metrics. State that predictions are saved
results, not live inference. Show one correct and one incorrect prediction.
The walkthrough needs no internet or Python installation.

## Simple answers if asked

**What was cleaned?** I validated files and labels, checked duplicates and
standardized image format and size. Nothing needed to be removed.

**Why 43 classes when the website says 42?** The checked-out annotations contain
IDs 0 through 42. I used the actual files and documented the mismatch.

**Can this recognize everything inside a fridge?** Not yet. These are whole-image
class labels; identifying multiple ingredients needs detection or multi-label data.

**Why compare these models?** ResNet is a familiar pretrained CNN baseline.
YOLO26n-cls is another pretrained classification option; we can compare actual
performance on our data instead of assuming one is better.

**What is macro F1?** It averages a score across categories so common ingredients
don't hide poor results on less common ones.

**How is this different from a complete recipe generator?** Ingredient recognition
is one component. We still need confirmed inventory, recipe matching and complete
recipe instructions.

**What did you personally do?** Dataset preparation, its audit and scripts, and the
new model comparison included in this folder. Other team branches are separate.

## Before presenting

- Confirm who is presenting each section; I should only claim my own work.
- Use MY-SPEAKING-SCRIPT.txt for the short update with measured model results.
- Agree on the next owner and deliverable rather than assigning teammates myself.
- Keep the shared slide deck unchanged until the team agrees what to include.
