import config
import re
from thefuzz import process, fuzz
import unicodedata

# ---------------------------------------------------------------------------
# Module-level compiled regexes
# ---------------------------------------------------------------------------

_DIR_RE = re.compile(r"\b(north|south|east|west)\b", re.IGNORECASE)

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
_JUNK_PATTERNS = re.compile(
    r"\b(if|where|find|go|you|your|it\s+and|continued|locale|destination|"
    r"steps?\s+away|close\s+the|get\s+there|east\s+of|west\s+of|north\s+of|south\s+of|"
    r"its\s+north|its\s+south|its\s+east|its\s+west|"
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
    """
    # Strategy 1: "at the X <hard boundary>"
    m = _AT_THE_RE.search(text)
    if m:
        return m.group(1).strip()

    # Strategy 2: last anchor verb → grab what follows
    anchors = list(_ANCHOR_RE.finditer(text))
    if anchors:
        start = anchors[-1].end()
        span = text[start:].strip()
    else:
        # Strategy 3: last occurrence of "the" as a weak anchor
        the_hits = list(re.finditer(r"\bthe\b", text, re.IGNORECASE))
        span = text[the_hits[-1].start():].strip() if the_hits else text.strip()

    # Trim at the first stop word that isn't at position 0
    stop_m = _STOPS.search(span)
    if stop_m and stop_m.start() > 0:
        span = span[:stop_m.start()].strip()

    return span if span else None


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

        # 3. Partial/embedded trigger match
        for trigger, group in self.text_lookup.items():
            if re.search(rf"\b{re.escape(trigger)}\b", text_lower):
                return group

        # 4. THE FUZZY SAFETY NET (The missing piece!)
        # If we get here, it's either a typo or a complex phrase.
        # We call the normalization function to snap it to the closest group.
        fuzzy_group = normalize_landmark_category(text_lower)
        
        if fuzzy_group in self.group_names:
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
        direction is one of 'N', 'S', 'E', 'W' or None.
        noun is None when no valid landmark could be extracted.
    """
    text = _normalize(text)

    # --- Direction ---
    direction = None
    dm = _DIR_RE.search(text)
    if dm:
        direction = dm.group(1).upper()[0]

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
        # Last-ditch rescue: take first two words (e.g. "music venue a few steps away" → "music venue")
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

def normalize_landmark_category(extracted_noun: str, threshold: int = 80) -> str:
    """
    Snaps a raw noun to the closest canonical LANDMARK_GROUPS key via fuzzy match.

    Example: 'musuem' -> 'MUSEUM', 'entrence' -> 'ENTRANCE'
    Returns the original (uppercased) string if no match clears the threshold.
    """
    canonical_keys = list(config.LANDMARK_GROUPS.keys())
    query = extracted_noun.strip().upper()

    if query in canonical_keys:
        return query

    best_match, score = process.extractOne(query, canonical_keys, scorer=fuzz.ratio)
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