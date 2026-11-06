import random
from pathlib import Path
import pandas as pd

random.seed(42)

ASPECTS = {
    "Camera": ["camera", "photo quality", "pictures", "lens", "selfie camera"],
    "Battery": ["battery", "battery life", "charging", "battery backup", "power backup"],
    "Display": ["display", "screen", "brightness", "resolution", "panel"],
    "Price": ["price", "cost", "value", "pricing", "budget"],
}

PHRASES = {
    "positive": {
        "Camera": ["is excellent", "takes amazing photos", "is very clear", "performs really well", "is impressive", "takes sharp pictures"],
        "Battery": ["lasts all day", "is excellent", "has great backup", "charges quickly", "is reliable", "lasts a long time"],
        "Display": ["looks beautiful", "is bright and clear", "has excellent colors", "is sharp", "looks fantastic", "is very good"],
        "Price": ["is reasonable", "is affordable", "offers great value", "is worth the money", "is budget friendly", "is quite fair"],
    },
    "negative": {
        "Camera": ["is poor", "takes blurry photos", "is disappointing", "performs badly", "is not good", "takes dull pictures"],
        "Battery": ["drains quickly", "is terrible", "has poor backup", "charges very slowly", "does not last long", "is disappointing"],
        "Display": ["is dull", "looks washed out", "is disappointing", "has poor brightness", "is not sharp", "looks bad"],
        "Price": ["is too high", "is expensive", "is overpriced", "is not worth the money", "is unreasonable", "is costly"],
    },
    "neutral": {
        "Camera": ["is average", "is okay", "is acceptable", "is neither good nor bad", "is ordinary", "works as expected"],
        "Battery": ["is average", "is okay", "is acceptable", "is neither good nor bad", "is ordinary", "works as expected"],
        "Display": ["is average", "is okay", "is acceptable", "is neither good nor bad", "is ordinary", "works as expected"],
        "Price": ["is average", "is okay", "is acceptable", "is neither cheap nor expensive", "is ordinary", "is as expected"],
    },
}

PREFIXES = ["", "overall, ", "for daily use, ", "in my experience, ", "honestly, ", "after using it, "]
SUFFIXES = ["", ".", " for this product.", " in everyday use.", " compared with similar products."]

def make_sentence(aspect, sentiment):
    noun = random.choice(ASPECTS[aspect])
    opinion = random.choice(PHRASES[sentiment][aspect])
    return f"{random.choice(PREFIXES)}the {noun} {opinion}{random.choice(SUFFIXES)}".strip()

rows = []
# Balanced synthetic teaching dataset: 4 aspects x 3 classes x 180 examples.
for aspect in ASPECTS:
    for sentiment in PHRASES:
        for _ in range(180):
            rows.append({"text": make_sentence(aspect, sentiment), "aspect": aspect, "sentiment": sentiment})

random.shuffle(rows)
out = Path(__file__).parent / "data" / "reviews.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved {len(rows)} rows to {out}")
