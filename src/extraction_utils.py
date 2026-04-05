import spacy
import config
import re
from thefuzz import process, fuzz

# Load the model once at the module level
nlp = spacy.load("en_core_web_sm")

class CategoricalMatcher:
    def __init__(self):
        # We use the TEXT_TO_GROUP_MAP from config
        # This maps "synagogue" -> "CHURCH", "pub" -> "BAR", etc.
        self.text_lookup = config.TEXT_TO_GROUP_MAP
        
        # We also want the Groups themselves to be triggers 
        # (e.g., if the text says "Meet at the BANK", it should trigger "BANK")
        self.group_names = {k.lower(): k for k in config.LANDMARK_GROUPS.keys()}

    def get_category(self, text):
        if not text:
            return "UNKNOWN"
        
        text_lower = text.lower().strip()
        
        # 1. Check if the word is a synonym in our Map (e.g., "deli" -> "SHOP")
        if text_lower in self.text_lookup:
            return self.text_lookup[text_lower]
        
        # 2. Check if the word is the Group Name itself (e.g., "pharmacy" -> "PHARMACY")
        if text_lower in self.group_names:
            return self.group_names[text_lower]
        
        # 3. Partial Match for compound brands (e.g., "Starbucks" contains "cafe" logic)
        # We check if any of our trigger words exist inside the extracted text
        for trigger, group in self.text_lookup.items():
            if re.search(rf"\b{re.escape(trigger)}\b", text_lower):
                return group
                
        return "UNKNOWN"

# Instantiate the matcher
matcher = CategoricalMatcher()


def extract_rvs_target(text: str) -> tuple:
    """
    Unified Extractor: Captures full brand spans, ignores directional fluff, 
    and clips at spatial boundaries (on, at the corner, etc.)
    """
    # 1. Standardize
    text_clean = text.replace("’", "'").replace(" ,", ",")

    # 2. Primary Anchor Search
    anchor_pattern = r"\b(at|me at|is at|to|the)\b\s+(.*)"
    match = re.search(anchor_pattern, text_clean, re.IGNORECASE)
    if not match: return "UNKNOWN", "UNKNOWN"
    
    span = match.group(2)

    # 3. Priority Jump (Skip directional fluff)
    # Jump ONLY if 'at' is followed by a landmark, not a spatial description
    jump_pattern = r"\bat\b\s+(?!(?:the\s+)?(?:corner|end|middle|side|south|north|west|east))\b(.*)"
    at_match = re.search(jump_pattern, span, re.IGNORECASE)
    if at_match:
        span = at_match.group(1)

    # 4. The Clipping Phase (Added 'at' to stops)
    stops = [
        r"\b(?:at|on|near|across|which|is|south|north|west|east|corner|end|middle)\b",
        r",", r"\."
    ]
    
    earliest_stop = len(span)
    for stop_pattern in stops:
        s_match = re.search(stop_pattern, span, re.IGNORECASE)
        if s_match and s_match.start() < earliest_stop:
            earliest_stop = s_match.start()
    
    noun = span[:earliest_stop].strip()

    # 5. POST-EXTRACTION CLEANUP (The "Tail" Fix)
    # Remove leading/trailing articles and dangling prepositions
    noun = re.sub(r"^(the|a|an)\s+", "", noun, flags=re.IGNORECASE)
    noun = re.sub(r"\s+(at|the|a|an)$", "", noun, flags=re.IGNORECASE) # Clean the tail

    # 6. Final Category Resolution
    category = matcher.get_category(noun)
    if category == "UNKNOWN":
        for word in noun.lower().split():
            word_cat = matcher.get_category(word)
            if word_cat != "UNKNOWN":
                category = word_cat
                break
                
    return category, noun.strip()


def normalize_landmark_category(extracted_noun, threshold=80):
    """
    Normalizes extracted nouns to the canonical keys in LANDMARK_GROUPS.
    Example: 'musuem' -> 'MUSEUM', 'entrence' -> 'ENTRANCE'
    """
    # 1. Get our "Source of Truth" keys from config
    canonical_keys = list(config.LANDMARK_GROUPS.keys())
    
    # 2. Pre-processing: Standardize to uppercase for matching
    query = extracted_noun.strip().upper()
    
    # 3. Quick Check: Is it already a perfect match?
    if query in canonical_keys:
        return query

    # 4. Fuzzy Match: Find the closest canonical key
    # process.extractOne returns (best_match, score)
    best_match, score = process.extractOne(query, canonical_keys, scorer=fuzz.ratio)
    
    if score >= threshold:
        # Success: We found a close enough match to justify 'correcting' it
        return best_match
    
    # 5. Failure: Too different, return original to avoid 'hallucinating' a category
    return query

def normalize_intent(extracted_noun, threshold=80):
    """
    Finalized Normalization Layer:
    - Snaps typos to Categories (Tier 1)
    - Passes Brands through to Name Search (Tier 2)
    """
    if not extracted_noun:
        return "UNKNOWN"
        
    canonical_keys = list(config.LANDMARK_GROUPS.keys())
    query = str(extracted_noun).strip().upper()
    
    # 1. Perfect Match
    if query in canonical_keys:
        return query
        
    # 2. Fuzzy Match (using WRatio for better singular/plural/case handling)
    best_match, score = process.extractOne(query, canonical_keys, scorer=fuzz.WRatio)
    
    if score >= threshold:
        return best_match
    
    # 3. PASS-THROUGH (For Brands like Starbucks, 7-Eleven, etc.)
    return query