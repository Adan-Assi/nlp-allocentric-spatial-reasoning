"""
extraction_utils.py
Rule-based NLP extraction from raw RVS instructions.

Returns: (category, noun, direction)
  - category : canonical LANDMARK_GROUPS key (e.g. "CAFE", "PARK") or "UNKNOWN" / "ROAD"
  - noun     : extracted noun phrase or None
  - direction: one of N, NE, E, SE, S, SW, W, NW, or None
               (8-way; intercardinals are NOT collapsed to N/S/E/W)

Improvements over the original repo version:
  1. Direction extraction is context-aware. A bare cardinal word like
     "East" in "East 49th Street" or "North Face" no longer triggers a
     direction. Direction is only extracted in valid grammatical frames
     (motion verb, "<dir> of", "on/to my <dir>", "<n> blocks <dir>").
  2. The 8-way direction is preserved instead of collapsed to its first
     letter. "northeast" → "NE", not "N". This is required for the
     project's allocentric-reasoning research question.
"""

import config
import re
import unicodedata

from thefuzz import process, fuzz


# ---------------------------------------------------------------------------
# Module-level compiled regexes
# ---------------------------------------------------------------------------

_DIR_WORD = r"(north|south|east|west|northeast|northwest|southeast|southwest)"

# Context-aware direction patterns. A cardinal word only counts as a
# direction when it appears in one of these grammatical frames. Everything
# else (e.g. "East Village", "North Face", "East 49th Street") is treated
# as part of a named entity, NOT as a direction signal.
#
# Semantics: the EARLIEST match across all patterns wins, preserving the
# "first directional phrase in the text" behavior of the old regex while
# filtering out false positives from capitalized direction-words in names.
_DIR_PATTERNS = [
    # 1. Motion verb + direction:  "walk north", "head northeast", "turn south"
    re.compile(
        r"\b(?:walk|head|go|travel|move|proceed|turn|face|continue)\s+"
        + _DIR_WORD + r"\b",
        re.IGNORECASE,
    ),
    # 2. Direction + of:  "north of the park", "southeast of the plaza"
    re.compile(r"\b" + _DIR_WORD + r"\s+of\b", re.IGNORECASE),
    # 3. (on|to) (the|my|your) <direction>:  "on my south", "to the north",
    #    "to your southwest", "on the northeast side"
    re.compile(
        r"\b(?:on|to)\s+(?:the|my|your)\s+" + _DIR_WORD + r"\b",
        re.IGNORECASE,
    ),
    # 4. <count> block(s) <direction>:  "2 blocks north", "a few blocks east",
    #    "a block west" (singular), "one block south"
    re.compile(
        r"\b(?:"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"a\s+few|several|couple\s+of|a\s+couple\s+of|a\s+couple)\s+blocks"  # plural
        r"|"
        r"(?:one|a)\s+block"  # singular ("a block", "one block")
        r")\s+" + _DIR_WORD + r"\b",
        re.IGNORECASE,
    ),
]

# Maps the 8 spelled-out directions to their canonical short forms.
_DIRECTION_MAP = {
    "north":     "N",
    "northeast": "NE",
    "east":      "E",
    "southeast": "SE",
    "south":     "S",
    "southwest": "SW",
    "west":      "W",
    "northwest": "NW",
}

_ANCHOR_RE = re.compile(
    r"\b(at|to|me\s+at|find\s+me\s+at|is\s+at|located\s+at|head\s+to|go\s+to|walk\s+to)\b",
    re.IGNORECASE,
)

# "at the X" shortcut — only when followed by a hard boundary
_AT_THE_RE = re.compile(
    r"\bat\s+the\s+([\w\s]{2,40}?)\s*(?=\s+on\b|\s+is\b|\s+at\b|\s+near\b|\s+just\b|\s+in\b|,|\.|\s+and\b|$)",
    re.IGNORECASE,
)

# Clause-fragment red flags: if the noun contains these, it's junk
#
# New entries (2026-04-25): "my <dir>" / "very <dir>" / "me and" — these are
# clause fragments that the anchor regex was leaking through as nouns
# ("my northwest", "very east", "me and …"), pushing many rows into
# Contradictory because the noun has no real referent.
_JUNK_PATTERNS = re.compile(
    r"\b(if|where|find|go|you|your|it\s+and|continued|locale|destination|"
    r"steps?\s+away|close\s+the|get\s+there|east\s+of|west\s+of|north\s+of|south\s+of|"
    r"its\s+north|its\s+south|its\s+east|its\s+west|"
    r"my\s+(?:north|south|east|west|northeast|northwest|southeast|southwest)|"
    r"very\s+(?:north|south|east|west|northeast|northwest|southeast|southwest)|"
    r"me\s+and|"
    r"most\s+north|most\s+south|most\s+east|most\s+west|"
    r"most\s+northern|most\s+southern|most\s+eastern|most\s+western|"
    r"northwest\s+of|northeast\s+of|southwest\s+of|southeast\s+of|"
    r"next\s+\w+\s+block|this\s+parking|pay\s+bike|train\s+tracks|"
    r"bridge\s+together|bridge\s+that|block\s+called|T-intersection|"
    r"least\s+two|freely\s+play|center\s+aisle)\b"
    r"|\.$",
    re.IGNORECASE,
)

# Stop patterns that terminate a noun phrase
_STOPS = re.compile(
    r"(?<!\w)\b(?:on|in|near|across|which|is|south|north|west|east|middle|"
    r"just|before|after|past|beside|behind|directly|close|towards|toward|"
    r"where|if|that|but|or|so|when|while|until|than|we'll|will|meet|"
    r"get|with|there|see|about)\b(?!\w)"
    r"|[,\.\!\?]",
    re.IGNORECASE,
)

# Noise suffixes to strip AFTER span extraction
_SUFFIX_NOISE = re.compile(
    r"\s+(?:directly|right|just|close|nearby|down|up|over|around|there|here|"
    r"a\s+few\s+steps?\s+\w+|close\s+to\s+the\s+road|next\s+to\s+\w+)\s*$",
    re.IGNORECASE,
)

# Leading articles
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


# Road/street references — routed to ROAD category instead of POI fuzzy search
_ROAD_SUFFIXES = re.compile(
    r"\b(street|st|avenue|ave|boulevard|blvd|drive|dr|road|rd|lane|ln|"
    r"place|pl|highway|hwy|route|circle|terrace)\b",
    re.IGNORECASE,
)

STOP_WORDS = frozenset({
    'the', 'a', 'an', 'and', 'meat', 'it', 'this', 'that',
    'here', 'there', 'somewhere', 'anywhere', 'place',
    # Pronouns / possessives that leaked through as standalone nouns when the
    # span anchor landed on a clause fragment. "its", "my", "me" alone are
    # never valid landmarks; previously these matched random POIs through
    # the snap fallback and produced false-positive Ambiguous/Answerable.
    'its', 'my', 'me',
})

MAX_NOUN_WORDS = 5  # spans longer than this are almost certainly clause fragments


# ---------------------------------------------------------------------------
# Internal helpers (module-level, not class methods)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Unicode-normalize and clean smart quotes/punctuation."""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u2019", "'").replace("\u2018", "'").replace(" ,", ",")


def _extract_span(text: str) -> str | None:
    """
    Pull the raw noun span from `text` using anchor → stop logic.
    Returns None if nothing plausible is found.

    Anchor-traversal order:
      1. `at the X <hard boundary>` — most reliable.
      2. Anchor verbs (`at`, `to`, `meet at`, …) tried in REVERSE so the
         last anchor is preferred (it usually points at the destination
         clause). If the last anchor lands at end-of-text or trims to
         empty after the stop-word cut, fall back to the next-earlier
         anchor — previously we'd just give up and return None, which
         was the dominant cause of "noun=None → Contradictory" rows.
      3. Last `the` as a weak anchor.
      4. The whole text, after stop-word trim.
    """
    def _trim_at_first_stop(s: str) -> str:
        stop_m = _STOPS.search(s)
        if stop_m and stop_m.start() > 0:
            return s[:stop_m.start()].strip()
        return s.strip()

    # Strategy 1: "at the X <hard boundary>"
    m = _AT_THE_RE.search(text)
    if m:
        return m.group(1).strip()

    # Strategy 2: anchor verbs, last-first with fallback.
    for anchor in reversed(list(_ANCHOR_RE.finditer(text))):
        cand = _trim_at_first_stop(text[anchor.end():].strip())
        if cand:
            return cand

    # Strategy 3: last "the" as a weak anchor.
    the_hits = list(re.finditer(r"\bthe\b", text, re.IGNORECASE))
    if the_hits:
        cand = _trim_at_first_stop(text[the_hits[-1].start():].strip())
        if cand:
            return cand

    # Strategy 4: whole text.
    cand = _trim_at_first_stop(text.strip())
    return cand if cand else None


# ---------------------------------------------------------------------------
# CategoricalMatcher
# ---------------------------------------------------------------------------

class CategoricalMatcher:
    """Maps extracted noun phrases to canonical LANDMARK_GROUPS categories."""

    def __init__(self):
        # "synagogue" -> "CHURCH", "pub" -> "BAR", etc.
        self.text_lookup = config.TEXT_TO_GROUP_MAP
        # Allows the group name itself as a trigger: "bank" -> "BANK"
        self.group_names = {k.lower(): k for k in config.LANDMARK_GROUPS.keys()}

    def get_category(self, text: str) -> str:
        if not text:
            return "UNKNOWN"

        text_lower = text.lower().strip()

        # 1. Exact synonym match (Fast)
        if text_lower in self.text_lookup:
            return self.text_lookup[text_lower]

        # 2. Group name match (Fast)
        if text_lower in self.group_names:
            return self.group_names[text_lower]

        # 3. Partial / embedded trigger match
        for trigger, group in self.text_lookup.items():
            if re.search(rf"\b{re.escape(trigger)}\b", text_lower):
                return group

        # 4. Fuzzy safety net (typo tolerance)
        # `normalize_landmark_category` returns an UPPERCASE canonical key
        # (e.g. "MUSEUM"); `self.group_names` keys are lowercase, so the
        # previous `fuzzy_group in self.group_names` was always False and
        # silently dropped every typo correction (e.g. "musuem" → "MUSEUM"
        # was found at score 83 but never returned). Check against
        # LANDMARK_GROUPS directly instead.
        fuzzy_group = normalize_landmark_category(text_lower)
        if fuzzy_group in config.LANDMARK_GROUPS:
            return fuzzy_group

        return "UNKNOWN"


# Singleton — instantiated once after the class definition
matcher = CategoricalMatcher()


# ---------------------------------------------------------------------------
# Primary public interface
# ---------------------------------------------------------------------------

def extract_rvs_target(text: str) -> tuple:
    """
    Extracts (category, noun, direction) from a raw RVS instruction.

    Hardened against:
    - clause fragments masquerading as nouns
    - trailing noise adverbs / prepositional phrases
    - street/road references (routed to ROAD category)
    - overly long spans that are clearly not landmark names

    Returns:
        (category: str, noun: str | None, direction: str | None)
        direction is one of 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW' or None.
        noun is None when no valid landmark could be extracted.
    """
    text = _normalize(text)

    # --- Direction ---
    # Find the earliest directional phrase across all grammatical frames.
    # A bare cardinal word (e.g. "East" in "East 49th Street") does NOT count.
    direction = None
    earliest = None
    for pat in _DIR_PATTERNS:
        m = pat.search(text)
        if m and (earliest is None or m.start() < earliest.start()):
            earliest = m
    if earliest:
        # Preserve full 8-way direction information.
        # Old behavior collapsed NE/NW -> N and SE/SW -> S, which lost
        # important allocentric information. We keep intercardinals intact.
        direction = _DIRECTION_MAP.get(earliest.group(1).lower())

    # --- Span extraction ---
    span = _extract_span(text)
    if span is None:
        return "UNKNOWN", None, direction

    # --- Strip leading article ---
    noun = _ARTICLE_RE.sub("", span).strip()

    # --- Strip trailing noise suffixes ---
    noun = _SUFFIX_NOISE.sub("", noun).strip()

    # --- Clause-fragment detection ---
    if _JUNK_PATTERNS.search(noun):
        return "UNKNOWN", None, direction

    # --- Length guard: more than MAX_NOUN_WORDS → almost certainly a clause ---
    if len(noun.split()) > MAX_NOUN_WORDS:
        # Last-ditch rescue: take first two words
        # (e.g. "music venue a few steps away" → "music venue")
        short = " ".join(noun.split()[:2])
        if not _JUNK_PATTERNS.search(short) and len(short) >= 3:
            noun = short
        else:
            return "UNKNOWN", None, direction

    # --- Stop word / too-short guard ---
    if noun.lower() in STOP_WORDS or len(noun) < 2:
        return "UNKNOWN", None, direction

    # --- Road suffix → ROAD category (avoids wasting POI fuzzy search on streets) ---
    if _ROAD_SUFFIXES.search(noun):
        return "ROAD", noun, direction

    # --- Category resolution ---
    try:
        category = matcher.get_category(noun)
    except Exception:
        category = "UNKNOWN"

    return category, noun, direction


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

def normalize_landmark_category(extracted_noun: str, threshold: int = 75) -> str:
    """
    Snaps a raw noun to the closest canonical LANDMARK_GROUPS key via fuzzy match.

    Examples:
      'musuem'           -> 'MUSEUM'      (single-word typo)
      'entrence'         -> 'ENTRANCE'    (single-word typo)
      'parking entrence' -> 'PARKING'     (multi-word with typo)

    Returns the original (uppercased) string if no match clears the threshold.

    Uses `fuzz.partial_ratio` (instead of plain `fuzz.ratio`) so multi-word
    inputs like "parking entrence" still match the single-token category
    "PARKING" — `ratio` would only score the full strings against each other
    and miss the embedded match. Threshold lowered from 80 to 75 to catch a
    few more single-token typos that were just below the previous bar.
    """
    canonical_keys = list(config.LANDMARK_GROUPS.keys())
    query = extracted_noun.strip().upper()

    if query in canonical_keys:
        return query

    best_match, score = process.extractOne(
        query, canonical_keys, scorer=fuzz.partial_ratio
    )
    return best_match if score >= threshold else query


def normalize_intent(extracted_noun: str, threshold: int = 80) -> str:
    """
    Two-tier normalization:
      Tier 1 — snaps typos to canonical categories (fuzzy match).
      Tier 2 — passes brands/unknowns through uppercased for name search.

    Example: 'Starbucks' won't match any category key and is passed through as-is.
    """
    if not extracted_noun:
        return "UNKNOWN"

    canonical_keys = list(config.LANDMARK_GROUPS.keys())
    query = str(extracted_noun).strip().upper()

    if query in canonical_keys:
        return query

    best_match, score = process.extractOne(query, canonical_keys, scorer=fuzz.WRatio)
    return best_match if score >= threshold else query
