import config
import re
from thefuzz import process, fuzz
import unicodedata
import numpy as np

# Semantic Matching (SOTA approach)
from sklearn.metrics.pairwise import cosine_similarity
from src.model_registry import get_embedding_model

# ---------------------------------------------------------------------------
# Module-level compiled regexes
# ---------------------------------------------------------------------------

# Attempt at fixing the 8 directions problem that an angle of 45 degrees didn't solve
_DIR_RE = re.compile(
    r"\b(northeast|northwest|southeast|southwest|north|south|east|west)\b",
    re.IGNORECASE,
)

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
    r"^(where|you|your)\b|"
    r"\b(if|find|go|it\s+and|continued|locale|destination|"
    r"steps?\s+away|close\s+the|get\s+there|east\s+of|west\s+of|north\s+of|south\s+of|"
    r"my\s+north|my\s+south|my\s+east|my\s+west|"
    r"its\s+north|its\s+south|its\s+east|its\s+west|"
    r"its\s+southwest|its\s+northwest|its\s+northeast|its\s+southeast|"
    r"most\s+north|most\s+south|most\s+east|most\s+west|"
    r"most\s+northern|most\s+southern|most\s+eastern|most\s+western|"
    r"northwest\s+of|northeast\s+of|southwest\s+of|southeast\s+of|"
    r"southeast\s+corner|northeast\s+corner|southwest\s+corner|northwest\s+corner|"
    r"corner\s+of\s+the|end\s+of\s+the|other\s+side\s+of|"
    r"south\s+part|north\s+part|east\s+part|west\s+part|"
    r"birthplace\s+of|entrance\s+to|"
    r"next\s+\w+\s+block|this\s+parking|pay\s+bike|train\s+tracks|"
    r"bridge\s+together|bridge\s+that|block\s+called|T-intersection|"
    r"least\s+two|freely\s+play|center\s+aisle)\b"
    r"|\.$",
    re.IGNORECASE,
)

# Stop patterns that terminate a noun phrase
_STOPS = re.compile(
    r"(?<!\w)\b(?:on|in|near|across|which|is|middle|"
    r"just|before|after|past|beside|behind|directly|close|towards|toward|"
    r"if|that|but|or|so|when|while|until|than|we'll|will|meet|"
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
    r"highway|hwy|route|circle|terrace)\b",
    re.IGNORECASE,
)

_FULL_DIR_MAP = {
    'northeast': 'NE', 'northwest': 'NW',
    'southeast': 'SE', 'southwest': 'SW',
    'north': 'N', 'south': 'S',
    'east': 'E', 'west': 'W',
}

STOP_WORDS = frozenset({
    'the', 'a', 'an', 'and', 'meat', 'it', 'this', 'that',
    'here', 'there', 'somewhere', 'anywhere', 'place',
})

MAX_NOUN_WORDS = 5  # spans longer than this are almost certainly clause fragments


# ---------------------------------------------------------------------------
# Semantic Matching — category descriptions for embedding
# ---------------------------------------------------------------------------
# Each category is described in natural language so the embedding model
# can match paraphrases like "doctor's office" → HOSPITAL,
# "place to work out" → GYM, "where you pray" → CHURCH.
# These descriptions are encoded ONCE at module load time.
#
# TEXT_TO_GROUP_MAP and LANDMARK_GROUPS in config.py are unchanged —
# they power the fast exact/partial matchers (steps 1-3).
# Embeddings only activate in step 4 when all fast steps fail!

# In HW1, we saw that word vectors are "known by the company they keep."
CATEGORY_DESCRIPTIONS = {
    "SHOP":       "shop store retail buying goods merchandise convenience supermarket grocery shopping",
    "PHARMACY":   "pharmacy drugstore medicine prescription chemist pills health drugs pick up prescriptions fill prescription",
    "BAR":        "bar pub tavern drinks alcohol beer nightclub brewery happy hour",
    "RESTAURANT": "restaurant diner eating food meal lunch dinner fast food sit down eating",
    "FOOD":       "pizza burger takeaway food court snacks fast food quick bite",
    "CAFE":       "cafe coffee shop bakery espresso latte pastry brunch morning coffee",
    "CLOTHES":    "clothing store fashion boutique apparel outfit dress shoes wear",
    "STORE":      "store department store big box retail shopping walmart target",
    "CHURCH":     "church synagogue mosque temple place of worship prayer religious service worship pray",
    "SCHOOL":     "school college university campus education learning academic study classes",
    "LIBRARY":    "library books reading public library branch lending borrow books read",
    "HOSPITAL":   "hospital clinic medical center emergency room doctor health treatment sick care",
    "OFFICE":     "office corporate headquarters workplace business professional work building",
    "PARK":       "park playground recreation ground outdoor green space sports field picnic",
    "GARDEN":     "garden botanical garden nature reserve green space flowers plants outdoor",
    "PARKING":    "parking lot garage car park parking space vehicle leave car",
    "MONUMENT":   "monument memorial fountain sculpture statue historic landmark art",
    "MUSEUM":     "museum gallery exhibition art history artifacts display",
    "THEATRE":    "theatre cinema movie arts performance concert hall show watch film",
    "MARKET":     "market marketplace farmers market street market stalls buy fresh food",
    "HOTEL":      "hotel motel inn hostel accommodation lodging stay sleep overnight",
    "BANK":       "bank ATM financial institution money withdraw deposit",
    "GYM":        "gym fitness center workout sports exercise yoga crossfit train work out lift weights",
    "ENTRANCE":   "entrance subway entrance building entrance metro entry door way in",
    "BENCH":      "bench seat rest area drinking water public seating sit down",
    "BIKE":       "citi bike bicycle parking bike station cycling rental rack lock bike",
    "RENTAL":     "car rental vehicle rental enterprise hertz avis scooter rent a car",
    "STATION":    "station pier dock terminal transport hub bus train subway",
    "POST":       "post office mailbox postal service mail send package stamp",
    "WATER":      "river lake pond waterfront water body stream waterway",
    "BUILDING":   "building apartment office residential commercial structure tower block",
}

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

    # Strip leading "where [you/I/they] <verb>" — these are clause fragments
    # but "noun where you <verb>" is a valid functional description (handled later)
    span = re.sub(r"^where\s+\w+\s+", "", span, flags=re.IGNORECASE).strip()

    # Trim at the first stop word that isn't at position 0
    stop_m = _STOPS.search(span)
    if stop_m and stop_m.start() > 0:
        span = span[:stop_m.start()].strip()

    return span if span else None


# ---------------------------------------------------------------------------
# CategoricalMatcher
# ---------------------------------------------------------------------------

class CategoricalMatcher:
    """
    Maps extracted noun phrases to canonical LANDMARK_GROUPS categories.

    Matching pipeline (ordered by speed, fast-exit on first hit):
      1. Exact match against TEXT_TO_GROUP_MAP           — O(1)
      2. Exact match against LANDMARK_GROUPS key names   — O(1)
      3. Partial/embedded trigger match                  — O(n)
      4. Semantic embedding similarity (SOTA, Lectures 1-2)
         Replaces fuzzy string match — handles synonyms,
         paraphrases, and context that string matching misses.
         e.g. "doctor's office" → HOSPITAL (not OFFICE)
              "place to work out" → GYM
              "where you get prescriptions" → PHARMACY
    """

    def __init__(self):
        # "synagogue" -> "CHURCH", "pub" -> "BAR", etc.
        self.text_lookup = config.TEXT_TO_GROUP_MAP
        # Allows the group name itself as a trigger: "bank" -> "BANK"
        self.group_names = {k.lower(): k for k in config.LANDMARK_GROUPS.keys()}

        # --- Semantic Matching Setup (Step 4) ---
        # Load model once: 'all-MiniLM-L6-v2' is small (80MB), fast,
        # and strong on short phrase similarity tasks.
        self._embedding_model = get_embedding_model()
        print("MODEL ID EXTRACTION:", id(self._embedding_model))

        # Pre-encode all category descriptions at init time — O(|categories|) once.
        # At query time, only the input noun is encoded — O(1) amortized.
        self._category_keys = list(CATEGORY_DESCRIPTIONS.keys())
        self._category_embeddings = self._embedding_model.encode(
            [CATEGORY_DESCRIPTIONS[k] for k in self._category_keys],
            normalize_embeddings=True,  # enables dot product as cosine similarity
            show_progress_bar=False,
        )
        print(f"✅ Semantic matcher ready with {len(self._category_keys)} categories.")


    def _semantic_match(self, text: str, threshold: float = 0.30) -> str:
        """
        Encode the input noun and find the most similar category description.
        Returns UNKNOWN if no category clears the similarity threshold.

        Threshold of 0.30 is conservative — embeddings are normalized so
        cosine similarity of 0.30 means meaningful semantic overlap.
        Lower = more permissive, Higher = more strict.
        """
        query_emb = self._embedding_model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        scores = (self._category_embeddings @ query_emb.T).flatten()
        
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        second_score = float(np.sort(scores)[-2])
        
        # Confident match: best score is meaningfully higher than runner-up
        # This handles cases where all scores are low but one is clearly best
        margin = best_score - second_score
        
        if best_score >= 0.30 or (best_score >= 0.20 and margin >= 0.05):
            return self._category_keys[best_idx]
        
        return "UNKNOWN"


    def get_category(self, text: str) -> str:
        if not text:
            return "UNKNOWN"

        text_lower = text.lower().strip()

        # 1. Exact synonym match (Fast) — TEXT_TO_GROUP_MAP
        if text_lower in self.text_lookup:
            return self.text_lookup[text_lower]

        # 2. Group name match (Fast) — LANDMARK_GROUPS keys
        if text_lower in self.group_names:
            return self.group_names[text_lower]

        # 3. Partial/embedded trigger match — O(n) over TEXT_TO_GROUP_MAP
        for trigger, group in self.text_lookup.items():
            if re.search(rf"\b{re.escape(trigger)}\b", text_lower):
                return group

        # 4. Semantic embedding similarity (SOTA replacement for fuzzy match)
        # Only reaches here when exact and partial matching both fail.
        # Handles: synonyms, paraphrases, context-dependent disambiguation.
        # Temporary debug inside step 4
        return self._semantic_match(text_lower)

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
    direction = None
    dm = _DIR_RE.search(text)
    if dm:
        direction = _FULL_DIR_MAP.get(dm.group(1).lower())

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