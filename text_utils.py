"""Shared helpers for normalizing ingredient / food-class names.

Both datasets have to speak the same vocabulary: the detector outputs class
names like "Tomatoes" or "green_pepper", and the recipe dataset lists
ingredients like "tomato" or "green peppers". Everything goes through
`normalize_ingredient` so the two can be joined.
"""
import re

# Words that end in "s" but are not plurals (or whose plural rule is irregular).
_NO_SINGULAR = {
    "asparagus", "hummus", "couscous", "molasses", "swiss", "brussels", "grits",
    "oats", "peas", "lentils", "chives", "greens", "noodles", "sprinkles",
    "bitters", "hollandaise", "jus", "citrus", "bass", "watercress", "series",
    "cookies", "chips", "glass", "anise", "mayonnaise", "quinoa", "salsa",
    "cheese", "rice", "juice", "sauce", "lettuce", "spice", "molasses", "dates",
}
_IRREGULAR = {
    "potatoes": "potato", "tomatoes": "tomato", "leaves": "leaf",
    "loaves": "loaf", "halves": "half", "knives": "knife", "mangoes": "mango",
    "cherries": "cherry", "berries": "berry", "anchovies": "anchovy",
    "radishes": "radish", "peaches": "peach", "sandwiches": "sandwich",
    "dishes": "dish", "boxes": "box", "tortillas": "tortilla",
}


def singularize(word: str) -> str:
    if word in _NO_SINGULAR or len(word) <= 3:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_ingredient(name: str) -> str:
    """Lowercase, replace separators, strip punctuation, singularize last word."""
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"[^a-z0-9 &']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    words = s.split(" ")
    words[-1] = singularize(words[-1])
    return " ".join(words)
