"""Clean the Roboflow "FridgeFoodDetection" dataset (YOLO export).

Input  (data/raw/fridge/):  data.yaml + {train,valid,test}/{images,labels}
Output (data/processed/fridge_yolo/): same layout, cleaned, with a fresh data.yaml
       data/processed/fridge_class_counts.csv
       data/processed/class_to_ingredient.csv   detector class -> Food.com ingredient
       reports/fridge_cleaning_report.md

Checks performed
  * image <-> label pairing (orphan labels dropped, unlabeled images reported)
  * unreadable / corrupt images dropped (needs Pillow)
  * label lines: 5 numbers, valid class id, box inside the image, non-zero size
    (boxes that poke slightly outside are clipped, degenerate ones dropped)
  * duplicate label lines removed
  * exact-duplicate images (md5) removed, including train/valid/test leakage
  * class names normalized so they match the recipe ingredient vocabulary

Run:  python src/clean_fridge.py --raw data/raw/fridge --out data/processed
"""
import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from text_utils import normalize_ingredient

try:
    from PIL import Image
except ImportError:  # still works without Pillow, just skips the corrupt-image check
    Image = None

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]
CLIP_TOL = 0.02  # boxes up to 2% outside the frame get clipped instead of dropped


def read_names(yaml_path: Path):
    text = yaml_path.read_text()
    try:
        import yaml
        names = yaml.safe_load(text)["names"]
        return list(names.values()) if isinstance(names, dict) else list(names)
    except ImportError:
        pass
    m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if m:
        return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
    names, grab = [], False
    for line in text.splitlines():
        if line.startswith("names:"):
            grab = True
            continue
        if grab:
            s = line.strip()
            if s.startswith("- "):
                names.append(s[2:].strip().strip("'\""))
            elif re.match(r"^\d+:", s):
                names.append(s.split(":", 1)[1].strip().strip("'\""))
            elif s:
                break
    return names


def md5(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_label_lines(lines, n_classes, stats):
    out, seen = [], set()
    for raw in lines:
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            # Polygon (segmentation) rows: convert to a bounding box
            if len(parts) >= 7 and len(parts) % 2 == 1:
                try:
                    c = int(float(parts[0]))
                    xs = [float(v) for v in parts[1::2]]
                    ys = [float(v) for v in parts[2::2]]
                except ValueError:
                    stats["bad_format"] += 1
                    continue
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                parts = [c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
                stats["polygon_to_box"] += 1
            else:
                stats["bad_format"] += 1
                continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            stats["bad_format"] += 1
            continue
        if not 0 <= c < n_classes:
            stats["bad_class_id"] += 1
            continue
        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        if min(x0, y0) < -CLIP_TOL or max(x1, y1) > 1 + CLIP_TOL:
            stats["box_out_of_frame"] += 1
            continue
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(1, x1), min(1, y1)
        w, h = x1 - x0, y1 - y0
        if w <= 1e-3 or h <= 1e-3:
            stats["degenerate_box"] += 1
            continue
        line = f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {w:.6f} {h:.6f}"
        if line in seen:
            stats["duplicate_box"] += 1
            continue
        seen.add(line)
        out.append(line)
    return out


def best_ingredient_match(cls_norm, vocab):
    """Exact match in the recipe vocab, else the most-used ingredient containing it."""
    if cls_norm in vocab:
        return cls_norm, vocab[cls_norm], "exact"
    cands = [(i, n) for i, n in vocab.items() if re.search(rf"\b{re.escape(cls_norm)}\b", i)]
    if cands:
        i, n = max(cands, key=lambda t: t[1])
        return i, n, "contains"
    return "", 0, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/fridge")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--vocab", default="data/processed/ingredient_vocab.csv")
    a = ap.parse_args()
    raw, out, rep = Path(a.raw), Path(a.out), Path(a.reports)
    dst = out / "fridge_yolo"
    if dst.exists():
        shutil.rmtree(dst)
    rep.mkdir(parents=True, exist_ok=True)

    names = read_names(raw / "data.yaml")
    names_norm = [normalize_ingredient(n) for n in names]
    log = ["# Fridge food detection – cleaning report", "",
           f"Classes in data.yaml: {len(names)}", ""]

    stats = Counter()
    split_counts = {}
    box_counts = defaultdict(Counter)  # split -> class -> boxes
    img_counts = defaultdict(Counter)  # split -> class -> images containing it
    seen_hash = {}

    for split in SPLITS:
        idir, ldir = raw / split / "images", raw / split / "labels"
        if not idir.exists():
            continue
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        imgs = {p.stem: p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT}
        labels = {p.stem: p for p in ldir.glob("*.txt")} if ldir.exists() else {}
        stats["orphan_labels"] += len(set(labels) - set(imgs))
        kept = 0
        for stem, ip in sorted(imgs.items()):
            stats["images_seen"] += 1
            if Image is not None:
                try:
                    with Image.open(ip) as im:
                        im.verify()
                except Exception:
                    stats["corrupt_image"] += 1
                    continue
            h = md5(ip)
            if h in seen_hash:
                key = "dup_image_cross_split" if seen_hash[h] != split else "dup_image_same_split"
                stats[key] += 1
                continue
            seen_hash[h] = split
            lines = labels[stem].read_text().splitlines() if stem in labels else []
            if stem not in labels:
                stats["image_without_label_file"] += 1
            clean = clean_label_lines(lines, len(names), stats)
            if not clean:
                stats["images_with_no_valid_boxes"] += 1
            shutil.copy2(ip, dst / split / "images" / ip.name)
            (dst / split / "labels" / f"{stem}.txt").write_text("\n".join(clean) + ("\n" if clean else ""))
            for c in [int(l.split()[0]) for l in clean]:
                box_counts[split][c] += 1
            for c in {int(l.split()[0]) for l in clean}:
                img_counts[split][c] += 1
            stats["boxes_kept"] += len(clean)
            kept += 1
        split_counts[split] = (len(imgs), kept)

    # data.yaml for the cleaned copy (YOLO / Ultralytics format)
    yaml_lines = ["path: .", "train: train/images", "val: valid/images", "test: test/images",
                  f"nc: {len(names)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names_norm)]
    (dst / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    # Per-class counts
    with open(out / "fridge_class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_raw", "class_norm", *[f"{s}_boxes" for s in SPLITS],
                    "total_boxes", "train_images"])
        for i, (n, nn) in enumerate(zip(names, names_norm)):
            row = [box_counts[s][i] for s in SPLITS]
            w.writerow([i, n, nn, *row, sum(row), img_counts["train"][i]])

    # Link detector classes to recipe ingredients
    vocab = {}
    if Path(a.vocab).exists():
        with open(a.vocab) as f:
            for r in csv.DictReader(f):
                vocab[r["ingredient"]] = int(r["n_recipes"])
    with open(out / "class_to_ingredient.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_norm", "recipe_ingredient", "n_recipes", "match_type"])
        matches = []
        for i, nn in enumerate(names_norm):
            ing, n, t = best_ingredient_match(nn, vocab) if vocab else ("", 0, "no_vocab")
            matches.append(t)
            w.writerow([i, nn, ing, n, t])

    # Report
    log += ["## Images per split (raw → kept)", "", "| split | raw | kept |", "|---|---|---|"]
    log += [f"| {s} | {r:,} | {k:,} |" for s, (r, k) in split_counts.items()]
    log += ["", "## Issues found and fixed", "", "| check | count |", "|---|---|"]
    log += [f"| {k} | {v:,} |" for k, v in sorted(stats.items())]
    totals = Counter()
    for s in SPLITS:
        totals.update(box_counts[s])
    log += ["", "## Class balance (all splits, boxes)", "", "| class | boxes |", "|---|---|"]
    log += [f"| {names_norm[c]} | {n:,} |" for c, n in totals.most_common()]
    missing = [names_norm[i] for i in range(len(names)) if totals[i] == 0]
    if missing:
        log += ["", f"Classes with zero boxes after cleaning: {', '.join(missing)}"]
    if vocab:
        mc = Counter(matches)
        log += ["", "## Link to recipe ingredients", "",
                f"exact: {mc['exact']}, partial: {mc['contains']}, unmatched: {mc['none']} "
                "(see data/processed/class_to_ingredient.csv)"]
    (rep / "fridge_cleaning_report.md").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
