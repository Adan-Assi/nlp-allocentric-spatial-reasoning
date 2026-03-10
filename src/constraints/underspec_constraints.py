import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set

# ----------------------------
# Constraint representation
# ----------------------------

@dataclass(frozen=True)
class Constraint:
    type: str              # "direction" | "radius" | "proximity" | "landmark"
    span: Tuple[int, int]  # character span in original text (start, end)
    meta: Dict             # extracted info, e.g. {"dir": "north"} or {"meters": 200}


# ----------------------------
# Helper: normalize whitespace after masking
# ----------------------------

def _clean_text(text: str) -> str:
    # collapse multiple spaces and fix spacing around punctuation a bit
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


# ============================================================
#  A) EXTRACTORS (4 functions): direction, radius, proximity, landmark
# ============================================================

# 1) Direction extractor
_DIR_PATTERNS = [
    # cardinal words
    (re.compile(r"\b(north|south|east|west)\b", re.IGNORECASE), "cardinal"),
    # compound like "northwest"
    (re.compile(r"\b(northwest|northeast|southwest|southeast)\b", re.IGNORECASE), "intercardinal"),
    # phrases like "to the north of"
    (re.compile(r"\b(to the|towards the|heading)\s+(north|south|east|west)\b", re.IGNORECASE), "phrase"),
    (re.compile(r"\b(north|south|east|west)\s+of\b", re.IGNORECASE), "of"),
]

def extract_direction(text: str) -> List[Constraint]:
    out: List[Constraint] = []
    for pat, kind in _DIR_PATTERNS:
        for m in pat.finditer(text):
            # pick the direction token from group(s)
            # try last group if exists else whole match
            dir_word = None
            if m.lastindex:
                dir_word = m.group(m.lastindex)
            else:
                dir_word = m.group(0)
            out.append(
                Constraint(
                    type="direction",
                    span=(m.start(), m.end()),
                    meta={"dir": dir_word.lower(), "kind": kind, "text": m.group(0)},
                )
            )
    return out


# 2) Radius extractor
# We convert "blocks" and "minutes" into meters with rough heuristics.
# These heuristics are OK for v1; you can tune later per city.
_BLOCK_TO_METERS = 80.0       # very rough (NYC blocks vary). Use 80m as a starter.
_MINUTE_TO_METERS = 80.0      # walking speed ~1.3 m/s => 78m/min; use 80m.

_RADIUS_PATTERNS = [
    # explicit meters / km
    re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*(m|meter|meters|km|kilometer|kilometers)\b", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(m|meter|meters|km|kilometer|kilometers)\b", re.IGNORECASE),

    # blocks
    re.compile(r"\b(\d+)\s+blocks?\b", re.IGNORECASE),
    re.compile(r"\b(a couple of|couple of)\s+blocks?\b", re.IGNORECASE),
    re.compile(r"\b(a few|few)\s+blocks?\b", re.IGNORECASE),

    # minutes walk
    re.compile(r"\b(\d+)\s+minutes?\s+(walk|walking)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+min\s+(walk|walking)\b", re.IGNORECASE),
]

def _to_meters(num: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ["m", "meter", "meters"]:
        return float(num)
    if unit in ["km", "kilometer", "kilometers"]:
        return float(num) * 1000.0
    raise ValueError(f"Unknown unit: {unit}")

def extract_radius(text: str) -> List[Constraint]:
    out: List[Constraint] = []

    for pat in _RADIUS_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)

            meters: Optional[float] = None
            kind: str = "unknown"

            # meters/km
            if m.lastindex and len(m.groups()) >= 2 and m.group(2).lower() in ["m","meter","meters","km","kilometer","kilometers"]:
                meters = _to_meters(float(m.group(1)), m.group(2))
                kind = "metric"

            # numeric blocks
            elif re.search(r"blocks?", raw, re.IGNORECASE) and re.search(r"\d", raw):
                n_blocks = float(re.search(r"\d+", raw).group(0))
                meters = n_blocks * _BLOCK_TO_METERS
                kind = "blocks_numeric"

            # couple/few blocks
            elif re.search(r"blocks?", raw, re.IGNORECASE):
                if re.search(r"couple", raw, re.IGNORECASE):
                    meters = 2.0 * _BLOCK_TO_METERS
                    kind = "blocks_couple"
                elif re.search(r"few", raw, re.IGNORECASE):
                    meters = 3.0 * _BLOCK_TO_METERS
                    kind = "blocks_few"

            # minutes walk
            elif re.search(r"(walk|walking)", raw, re.IGNORECASE) and re.search(r"\d", raw):
                mins = float(re.search(r"\d+", raw).group(0))
                meters = mins * _MINUTE_TO_METERS
                kind = "minutes_walk"

            if meters is None:
                continue

            out.append(
                Constraint(
                    type="radius",
                    span=(m.start(), m.end()),
                    meta={"meters": float(meters), "kind": kind, "text": raw},
                )
            )

    return out


# 3) Proximity extractor
# We treat proximity as a small-radius constraint (v1 heuristic).
# Later you can split "across from" vs "near" vs "next to".
_PROX_PATTERNS = [
    re.compile(r"\b(near|nearby|close to|next to|beside|by)\b", re.IGNORECASE),
    re.compile(r"\b(across from|across the street from|across the street)\b", re.IGNORECASE),
    re.compile(r"\b(adjacent to)\b", re.IGNORECASE),
]

def extract_proximity(text: str) -> List[Constraint]:
    out: List[Constraint] = []
    for pat in _PROX_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0).lower()

            # heuristic meters
            if "across" in phrase:
                meters = 30.0
                kind = "across"
            elif "next to" in phrase or "adjacent" in phrase or "beside" in phrase:
                meters = 20.0
                kind = "adjacent"
            else:
                meters = 80.0
                kind = "near"

            out.append(
                Constraint(
                    type="proximity",
                    span=(m.start(), m.end()),
                    meta={"meters": meters, "kind": kind, "text": m.group(0)},
                )
            )
    return out


# 4) Landmark extractor
#    Detects named places, street references, and waypoint phrases in RVS instructions.
#    Categories:
#      - "past the X" / "passing X"           → intermediate waypoint
#      - "at the corner of X and Y"           → intersection reference
#      - "on East 49th Street" / "on Broadway"→ street reference
#      - Proper-noun sequences (2+ words)     → named places (Central Park, Duane Reade)
#      - Common POI nouns                     → generic landmarks (cafe, restaurant, church)

_LANDMARK_PATTERNS = [
    # "past the library", "past Central Park", "passing the museum"
    (re.compile(
        r"\bpast(?:ing)?\s+(?:the\s+)?[A-Za-z][A-Za-z0-9'\s-]{2,30}?(?=[,.\n]|\band\b|\bto\b|\bon\b|$)",
        re.IGNORECASE,
    ), "waypoint"),

    # "at the corner of 5th and 42nd", "at the corner of Broadway and Houston"
    (re.compile(
        r"\bat\s+the\s+corner\s+of\s+[^,.]{5,50}",
        re.IGNORECASE,
    ), "intersection"),

    # Street references: "on East 49th Street", "on Broadway", "on 2nd Avenue"
    (re.compile(
        r"\bon\s+(?:(?:East|West|North|South)\s+)?\d+(?:st|nd|rd|th)\s+(?:Street|Avenue|Ave|St|Place|Pl|Boulevard|Blvd|Drive|Dr|Road|Rd)",
        re.IGNORECASE,
    ), "street_numbered"),

    # Named street references: "on Broadway", "on FDR Drive", "on Greenwich Street"
    (re.compile(
        r"\bon\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Street|Avenue|Ave|St|Place|Pl|Boulevard|Blvd|Drive|Dr|Road|Rd|Way|Lane|Ln)",
        re.IGNORECASE,
    ), "street_named"),

    # Proper-noun sequences: "Central Park", "Empire State Building", "MacDougal-Sullivan Gardens"
    # Must be 2+ capitalized words, min 5 chars total
    (re.compile(
        r"\b(?:[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s+){1,4}(?:[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?)\b",
    ), "proper_noun"),

    # Common POI type nouns that appear in RVS instructions
    (re.compile(
        r"\b(?:the\s+)?(?:cafe|restaurant|church|park|garden|museum|gallery|library|fountain|"
        r"school|hospital|pharmacy|theater|theatre|synagogue|mosque|cathedral|"
        r"station|subway|bodega|deli|bakery|bar|pub|hotel|hostel|"
        r"bench|playground|plaza|market|supermarket|grocery|"
        r"bicycle\s+(?:parking|rental)|bike\s+(?:parking|rental)|"
        r"parking\s+(?:lot|garage|entrance)|vending\s+machine|"
        r"dog\s+park|fire\s+station|post\s+office|"
        r"tobacco\s+shop|gift\s+shop|hardware\s+shop|clothes\s+shop)\b",
        re.IGNORECASE,
    ), "poi_noun"),
]

# Words to EXCLUDE from proper-noun matches (common English words that happen to be capitalized)
_PROPER_NOUN_BLACKLIST = {
    "meet", "head", "go", "walk", "turn", "continue", "follow", "cross",
    "you", "your", "the", "this", "that", "there", "just", "right", "left",
    "it", "its", "can", "get", "keep", "take", "make", "find", "look",
    "about", "after", "before", "between", "from", "into", "onto", "with",
    "street", "avenue", "block", "blocks", "side", "corner", "end",
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
}


def extract_landmark(text: str) -> List[Constraint]:
    out: List[Constraint] = []
    seen_spans: set = set()  # avoid duplicate overlapping matches

    for pat, kind in _LANDMARK_PATTERNS:
        for m in pat.finditer(text):
            # Skip if this span overlaps with an already-found landmark
            span = (m.start(), m.end())
            if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                continue

            matched_text = m.group(0).strip()

            # For proper nouns, filter out blacklisted words
            if kind == "proper_noun":
                words = matched_text.split()
                if len(words) < 2:
                    continue
                # Check that at least one word is NOT blacklisted
                real_words = [w for w in words if w.lower() not in _PROPER_NOUN_BLACKLIST]
                if len(real_words) < 1:
                    continue

            # For POI nouns, strip leading "the "
            if kind == "poi_noun" and matched_text.lower().startswith("the "):
                matched_text = matched_text[4:]

            out.append(
                Constraint(
                    type="landmark",
                    span=span,
                    meta={"kind": kind, "text": matched_text},
                )
            )
            seen_spans.add(span)

    return out


# ============================================================
#  B) ORCHESTRATOR: choose which constraints to extract
# ============================================================

EXTRACTORS = {
    "direction": extract_direction,
    "radius": extract_radius,
    "proximity": extract_proximity,
    "landmark": extract_landmark,
}

def extract_constraints(text: str, enabled: Iterable[str] = ("direction", "radius", "proximity", "landmark")) -> List[Constraint]:
    constraints: List[Constraint] = []
    for name in enabled:
        if name not in EXTRACTORS:
            raise ValueError(f"Unknown constraint type: {name}")
        constraints.extend(EXTRACTORS[name](text))
    # Sort by span start for stable behavior
    constraints.sort(key=lambda c: (c.span[0], c.span[1]))
    return constraints


# ============================================================
#  C) MASKERS (4 functions): remove phrases for each type
# ============================================================

def mask_direction_phrases(text: str) -> str:
    # remove direction keywords/phrases
    # Keep it conservative: remove common direction chunks but not everything
    patterns = [
        r"\b(to the|towards the|heading)\s+(north|south|east|west)\b",
        r"\b(north|south|east|west)\s+of\b",
        r"\b(northwest|northeast|southwest|southeast)\b",
        r"\b(north|south|east|west)\b",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, "[MASK_DIR]", out, flags=re.IGNORECASE)
    return _clean_text(out)

def mask_radius_phrases(text: str) -> str:
    patterns = [
        r"\bwithin\s+\d+(?:\.\d+)?\s*(m|meter|meters|km|kilometer|kilometers)\b",
        r"\b\d+(?:\.\d+)?\s*(m|meter|meters|km|kilometer|kilometers)\b",
        r"\b\d+\s+blocks?\b",
        r"\b(a couple of|couple of)\s+blocks?\b",
        r"\b(a few|few)\s+blocks?\b",
        r"\b\d+\s+(minutes?\s+(walk|walking)|min\s+(walk|walking))\b",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, "[MASK_RAD]", out, flags=re.IGNORECASE)
    return _clean_text(out)

def mask_proximity_phrases(text: str) -> str:
    patterns = [
        r"\b(near|nearby|close to|next to|beside|by)\b",
        r"\b(across from|across the street from|across the street)\b",
        r"\b(adjacent to)\b",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, "[MASK_PROX]", out, flags=re.IGNORECASE)
    return _clean_text(out)

def mask_landmark_phrases(text: str) -> str:
    """
    Masks landmark references: 'past the X', street names, proper-noun places, POI nouns.
    Replaces with [MASK_LM] to indicate removed landmark info.
    """
    patterns = [
        # "past the library", "passing Central Park"
        r"\bpast(?:ing)?\s+(?:the\s+)?[A-Za-z][A-Za-z0-9'\s-]{2,30}?(?=[,.\n]|\band\b|\bto\b|\bon\b|$)",

        # "at the corner of X and Y"
        r"\bat\s+the\s+corner\s+of\s+[^,.]{5,50}",

        # Numbered streets: "on East 49th Street", "on 2nd Avenue"
        r"\bon\s+(?:(?:East|West|North|South)\s+)?\d+(?:st|nd|rd|th)\s+(?:Street|Avenue|Ave|St|Place|Pl|Boulevard|Blvd|Drive|Dr|Road|Rd)",

        # Named streets: "on Broadway", "on Greenwich Street"
        r"\bon\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Street|Avenue|Ave|St|Place|Pl|Boulevard|Blvd|Drive|Dr|Road|Rd|Way|Lane|Ln)",

        # Common POI nouns
        (r"\b(?:the\s+)?(?:cafe|restaurant|church|park|garden|museum|gallery|library|fountain|"
         r"school|hospital|pharmacy|theater|theatre|synagogue|mosque|cathedral|"
         r"station|subway|bodega|deli|bakery|bar|pub|hotel|hostel|"
         r"bench|playground|plaza|market|supermarket|grocery|"
         r"bicycle\s+(?:parking|rental)|bike\s+(?:parking|rental)|"
         r"parking\s+(?:lot|garage|entrance)|vending\s+machine|"
         r"dog\s+park|fire\s+station|post\s+office|"
         r"tobacco\s+shop|gift\s+shop|hardware\s+shop|clothes\s+shop)\b"),
    ]
    out = text
    for p in patterns:
        out = re.sub(p, "[MASK_LM]", out, flags=re.IGNORECASE)
    return _clean_text(out)


MASKERS = {
    "direction": mask_direction_phrases,
    "radius": mask_radius_phrases,
    "proximity": mask_proximity_phrases,
    "landmark": mask_landmark_phrases,
}

def apply_masks(text: str, drop_types: Iterable[str]) -> str:
    out = text
    for t in drop_types:
        if t not in MASKERS:
            raise ValueError(f"Unknown constraint type for masking: {t}")
        out = MASKERS[t](out)
    return _clean_text(out)


# ============================================================
#  D) Variant generator: choose which constraints to drop
# ============================================================

def generate_variants_for_text(
    text: str,
    enabled_types: Iterable[str] = ("direction", "radius", "proximity", "landmark"),
    drop_sets: Optional[List[Set[str]]] = None,
) -> List[Dict]:
    """
    Returns list of dicts:
      {
        "variant_text": ...,
        "kept_types": [...],
        "dropped_types": [...],
      }

    With 4 types, generates up to 2^4 - 1 = 15 masked variants + 1 original = 16 total.
    """
    enabled_types = list(enabled_types)

    if drop_sets is None:
        # default: all non-empty subsets of enabled_types
        drop_sets = []
        n = len(enabled_types)
        for mask in range(1, 1 << n):
            drop = {enabled_types[i] for i in range(n) if (mask & (1 << i))}
            drop_sets.append(drop)

    variants = []

    # Always include original (drop nothing)
    variants.append({
        "variant_text": _clean_text(text),
        "kept_types": enabled_types,
        "dropped_types": [],
    })

    for drop in drop_sets:
        kept = [t for t in enabled_types if t not in drop]
        vtext = apply_masks(text, drop)
        variants.append({
            "variant_text": vtext,
            "kept_types": kept,
            "dropped_types": sorted(list(drop)),
        })

    return variants
