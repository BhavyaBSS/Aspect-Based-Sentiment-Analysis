import re

ASPECT_KEYWORDS = {
    "Camera": ["camera", "photo", "photos", "picture", "pictures", "lens", "photography", "selfie"],
    "Battery": ["battery", "charging", "charge", "charger", "backup", "power"],
    "Display": ["display", "screen", "brightness", "resolution", "panel"],
    "Price": ["price", "cost", "expensive", "cheap", "affordable", "overpriced", "value", "budget"],
}

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.!?,' -]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def split_clauses(text):
    text = clean_text(text)
    # Contrast words often separate different aspect opinions.
    parts = re.split(r"[.!?;]+|\b(?:but|however|although|though|while|whereas)\b", text)
    return [p.strip(" ,") for p in parts if p.strip(" ,")]

def detect_aspects(text):
    low = clean_text(text)
    found = []
    for aspect, words in ASPECT_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", low) for word in words):
            found.append(aspect)
    return found

def extract_aspect_contexts(text):
    clauses = split_clauses(text)
    result = {}
    for aspect, words in ASPECT_KEYWORDS.items():
        matches = []
        for clause in clauses:
            if any(re.search(rf"\b{re.escape(word)}\b", clause) for word in words):
                matches.append(clause)
        if matches:
            result[aspect] = " . ".join(matches)
    return result
