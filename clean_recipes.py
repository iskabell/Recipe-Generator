"""Clean the Food.com Recipes and Interactions dataset (Kaggle: shuyangli94).

Input  (data/raw/foodcom/):  RAW_recipes.csv, RAW_interactions.csv
Output (data/processed/):
    recipes_clean.csv.gz        full cleaned recipes (lists stored as JSON)
    recipes_slim.csv.gz         compact version for the recommender
                                (no steps/description)
    interactions_clean.csv.gz   cleaned user-recipe ratings
    reviews_text.csv.gz         review text (large; gitignored, regenerate locally)
    ingredient_vocab.csv        every normalized ingredient + recipe count
    reports/recipes_cleaning_report.md

Run:  python src/clean_recipes.py --raw data/raw/foodcom --out data/processed
"""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from text_utils import normalize_ingredient

NUTRITION_COLS = [
    "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "saturated_fat_pdv", "carbs_pdv",
]
MAX_MINUTES = 2 * 24 * 60      # recipes longer than 2 days are almost all data errors
MAX_CALORIES = 5000            # per serving; above this is almost always a bad entry


def parse_list(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        v = ast.literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def clean_recipes(raw: Path, log: list) -> pd.DataFrame:
    df = pd.read_csv(raw / "RAW_recipes.csv")
    log.append(f"Loaded RAW_recipes.csv: {len(df):,} rows, columns: {', '.join(df.columns)}")

    def step(desc, mask):
        n = int((~mask).sum())
        log.append(f"- {desc}: removed {n:,}")
        return df[mask]

    # Missing / duplicate ids and names
    df = step("rows with missing id", df["id"].notna())
    df = step("duplicate recipe ids", ~df["id"].duplicated())
    df = step("rows with missing name", df["name"].notna() & (df["name"].astype(str).str.strip() != ""))
    df["name"] = df["name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    # Parse stringified Python lists
    for col in ["tags", "steps", "ingredients", "nutrition"]:
        df[col] = df[col].map(parse_list)

    df["ingredients"] = df["ingredients"].map(
        lambda xs: [str(i).strip().lower() for i in xs if str(i).strip()]
    )
    df["ingredients_norm"] = df["ingredients"].map(
        lambda xs: sorted({normalize_ingredient(i) for i in xs if normalize_ingredient(i)})
    )
    df["steps"] = df["steps"].map(lambda xs: [str(s).strip() for s in xs if str(s).strip()])
    df["tags"] = df["tags"].map(lambda xs: [str(t).strip().lower() for t in xs if str(t).strip()])

    df = step("recipes with no ingredients", df["ingredients"].map(len) > 0)
    df = step("recipes with no steps", df["steps"].map(len) > 0)
    df["n_ingredients"] = df["ingredients"].map(len)
    df["n_steps"] = df["steps"].map(len)

    # Cooking time outliers
    df = step(f"minutes <= 0 or > {MAX_MINUTES}", (df["minutes"] > 0) & (df["minutes"] <= MAX_MINUTES))

    # Nutrition -> columns
    df = step("malformed nutrition (not 7 values)", df["nutrition"].map(len) == 7)
    nut = pd.DataFrame(df["nutrition"].tolist(), columns=NUTRITION_COLS, index=df.index).astype(float)
    df = pd.concat([df.drop(columns=["nutrition"]), nut], axis=1)
    df = step(f"calories > {MAX_CALORIES} or negative nutrition",
              (df["calories"] <= MAX_CALORIES) & (nut.loc[df.index] >= 0).all(axis=1))

    # Near-duplicates: same name + same ingredient set
    key = df["name"].str.lower() + "|" + df["ingredients_norm"].map(lambda x: ",".join(x))
    df = step("duplicate (name, ingredients) recipes", ~key.duplicated())

    df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")
    df["id"] = df["id"].astype(int)
    log.append(f"Recipes after cleaning: {len(df):,}")
    return df


def clean_interactions(raw: Path, recipe_ids: set, log: list) -> pd.DataFrame:
    it = pd.read_csv(raw / "RAW_interactions.csv")
    log.append(f"\nLoaded RAW_interactions.csv: {len(it):,} rows")
    n0 = len(it)
    it = it.dropna(subset=["user_id", "recipe_id", "rating"])
    it = it[it["recipe_id"].isin(recipe_ids)]
    log.append(f"- missing keys or recipe removed during cleaning: removed {n0 - len(it):,}")
    it["date"] = pd.to_datetime(it["date"], errors="coerce")
    n0 = len(it)
    it = it.sort_values("date").drop_duplicates(["user_id", "recipe_id"], keep="last")
    log.append(f"- duplicate (user, recipe) pairs, kept latest: removed {n0 - len(it):,}")
    it["review"] = it["review"].fillna("").astype(str).str.strip()
    # On Food.com a 0 means "reviewed without a star rating", not a 0-star review.
    it["has_rating"] = it["rating"] > 0
    log.append(f"- interactions with rating 0 (review only, kept but flagged): {(~it['has_rating']).sum():,}")
    log.append(f"Interactions after cleaning: {len(it):,}")
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/foodcom")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()
    raw, out, rep = Path(a.raw), Path(a.out), Path(a.reports)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    log = ["# Food.com recipes – cleaning report", ""]
    recipes = clean_recipes(raw, log)
    inter = clean_interactions(raw, set(recipes["id"]), log)

    rated = inter[inter["has_rating"]]
    stats = rated.groupby("recipe_id")["rating"].agg(avg_rating="mean", n_ratings="count")
    recipes = recipes.merge(stats, left_on="id", right_index=True, how="left")
    recipes["n_ratings"] = recipes["n_ratings"].fillna(0).astype(int)
    recipes["avg_rating"] = recipes["avg_rating"].round(3)

    # Ingredient vocabulary (how many recipes use each normalized ingredient)
    cnt = Counter(i for xs in recipes["ingredients_norm"] for i in xs)
    vocab = pd.DataFrame(cnt.most_common(), columns=["ingredient", "n_recipes"])
    vocab.to_csv(out / "ingredient_vocab.csv", index=False)

    to_json = lambda s: s.map(json.dumps)
    full = recipes.copy()
    for c in ["tags", "steps", "ingredients", "ingredients_norm"]:
        full[c] = to_json(full[c])
    cols = ["id", "name", "description", "minutes", "n_steps", "steps", "n_ingredients",
            "ingredients", "ingredients_norm", "tags", *NUTRITION_COLS,
            "avg_rating", "n_ratings", "contributor_id", "submitted"]
    full[cols].to_csv(out / "recipes_clean.csv.gz", index=False, compression="gzip")

    slim_cols = ["id", "name", "minutes", "n_ingredients", "ingredients_norm",
                 "calories", "protein_pdv", "avg_rating", "n_ratings"]
    full[slim_cols].to_csv(out / "recipes_slim.csv.gz", index=False, compression="gzip")

    # Ratings and review text are split so each file stays under GitHub's 100 MB limit.
    inter[["user_id", "recipe_id", "date", "rating", "has_rating"]].to_csv(
        out / "interactions_clean.csv.gz", index=False, compression="gzip")
    inter[["user_id", "recipe_id", "review"]].to_csv(
        out / "reviews_text.csv.gz", index=False, compression="gzip")

    # Summary stats for the report
    log += ["", "## Summary", "",
            f"| | value |", "|---|---|",
            f"| recipes | {len(recipes):,} |",
            f"| interactions | {len(inter):,} |",
            f"| unique users | {inter['user_id'].nunique():,} |",
            f"| unique normalized ingredients | {len(vocab):,} |",
            f"| median minutes | {recipes['minutes'].median():.0f} |",
            f"| median ingredients / recipe | {recipes['n_ingredients'].median():.0f} |",
            f"| median calories | {recipes['calories'].median():.0f} |",
            f"| recipes with ≥1 rating | {(recipes['n_ratings'] > 0).sum():,} |",
            f"| mean star rating (rated only) | {rated['rating'].mean():.2f} |",
            "", "## Top 25 ingredients", "", "| ingredient | recipes |", "|---|---|"]
    log += [f"| {r.ingredient} | {r.n_recipes:,} |" for r in vocab.head(25).itertuples()]
    (rep / "recipes_cleaning_report.md").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
