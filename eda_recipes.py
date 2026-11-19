"""Quick look at the cleaned recipe data -> reports/recipes_eda.png"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

r = pd.read_csv("data/processed/recipes_slim.csv.gz")
it = pd.read_csv("data/processed/interactions_clean.csv.gz")
STAPLES = {"salt", "water", "pepper", "sugar", "salt and pepper", "black pepper"}
cnt = Counter(i for xs in r["ingredients_norm"].map(json.loads) for i in xs if i not in STAPLES)
top = pd.Series(dict(cnt.most_common(20)))

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
c = "#3b6ea8"
ax[0, 0].hist(r["minutes"].clip(upper=240), bins=48, color=c)
ax[0, 0].set(title="Cooking time (capped at 240 min)", xlabel="minutes", ylabel="recipes")
ax[0, 1].hist(r["n_ingredients"], bins=range(1, 35), color=c)
ax[0, 1].set(title="Ingredients per recipe", xlabel="ingredients", ylabel="recipes")
rated = it[it["has_rating"]]["rating"].value_counts().sort_index()
ax[1, 0].bar(rated.index.astype(str), rated.values, color=c)
ax[1, 0].set(title="Star ratings (rated interactions)", xlabel="stars", ylabel="count")
ax[1, 1].barh(top.index[::-1], top.values[::-1], color=c)
ax[1, 1].set(title="Top 20 ingredients (excl. salt/pepper/sugar/water)", xlabel="recipes")
for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Food.com after cleaning: {len(r):,} recipes, {len(it):,} interactions")
fig.tight_layout()
fig.savefig("reports/recipes_eda.png", dpi=120)
print("saved reports/recipes_eda.png")
