# Food.com recipes – cleaning report

Loaded RAW_recipes.csv: 231,637 rows, columns: name, id, minutes, contributor_id, submitted, tags, nutrition, n_steps, steps, description, ingredients, n_ingredients
- rows with missing id: removed 0
- duplicate recipe ids: removed 0
- rows with missing name: removed 1
- recipes with no ingredients: removed 0
- recipes with no steps: removed 1
- minutes <= 0 or > 2880: removed 1,776
- malformed nutrition (not 7 values): removed 0
- calories > 5000 or negative nutrition: removed 996
- duplicate (name, ingredients) recipes: removed 26
Recipes after cleaning: 228,837

Loaded RAW_interactions.csv: 1,132,367 rows
- missing keys or recipe removed during cleaning: removed 11,305
- duplicate (user, recipe) pairs, kept latest: removed 0
- interactions with rating 0 (review only, kept but flagged): 59,935
Interactions after cleaning: 1,121,062

## Summary

| | value |
|---|---|
| recipes | 228,837 |
| interactions | 1,121,062 |
| unique users | 224,522 |
| unique normalized ingredients | 13,648 |
| median minutes | 40 |
| median ingredients / recipe | 9 |
| median calories | 312 |
| recipes with ≥1 rating | 223,915 |
| mean star rating (rated only) | 4.66 |

## Top 25 ingredients

| ingredient | recipes |
|---|---|
| salt | 84,650 |
| butter | 54,166 |
| egg | 49,551 |
| onion | 48,321 |
| sugar | 43,542 |
| garlic clove | 35,135 |
| water | 34,312 |
| olive oil | 32,631 |
| flour | 25,794 |
| milk | 25,442 |
| pepper | 22,140 |
| brown sugar | 18,362 |
| garlic | 17,806 |
| all purpose flour | 17,380 |
| baking powder | 17,151 |
| salt and pepper | 15,349 |
| parmesan cheese | 14,739 |
| lemon juice | 14,045 |
| baking soda | 13,757 |
| vegetable oil | 13,720 |
| carrot | 13,533 |
| black pepper | 12,987 |
| vanilla | 12,926 |
| cinnamon | 12,308 |
| tomato | 11,860 |
