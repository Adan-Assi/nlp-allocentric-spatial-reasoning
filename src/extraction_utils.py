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
    Unified Extractor (v3): Uses right-most anchor matching to bypass 
    initial directional instructions (e.g., 'at the west') and isolate 
    the actual landmark name (e.g., 'at Dunkin'). 
    
    Clips the span at spatial boundaries (on, which is, etc.) and 
    normalizes the result into a (Category, Noun) tuple.
    
    Unified Extractor: Captures full brand spans, ignores directional fluff, 
    and clips at spatial boundaries (on, at the corner, etc.)
    """
    
    # 1. Clean
    text_clean = text.replace("’", "'").replace(" ,", ",")

    # 2. THE FIX: Find the LAST 'at' or 'to' before the landmark
    # This ignores "at the west" and finds "at Ben & Jerry's"
    potential_anchors = [m.start() for m in re.finditer(r"\b(at|to|me at|is at)\b", text_clean, re.IGNORECASE)]
    
    if not potential_anchors:
        return "UNKNOWN", "UNKNOWN"
    
    # We take the last anchor found in the sentence
    start_idx = potential_anchors[-1]
    # Move the pointer past the anchor word itself (e.g., skip 'at ')
    span = re.sub(r"^(at|to|me at|is at)\s+", "", text_clean[start_idx:], flags=re.IGNORECASE)

    # 3. Clipping (No 'at' or 'the' here!)
    stops = [
        # Spatial boundaries
        r"\b(?:on|near|across|which|is|south|north|west|east|corner|end|middle)\b",
        # Template boundaries (The "Fluff" Fix)
        r"\b(?:and|let's|lets|very|place|to|the)\b", 
        r",", r"\."
    ]
        
    earliest_stop = len(span)
    for stop_pattern in stops:
        s_match = re.search(stop_pattern, span, re.IGNORECASE)
        if s_match and s_match.start() < earliest_stop:
            earliest_stop = s_match.start()
    
    noun = span[:earliest_stop].strip()

    # 4. Cleanup
    noun = re.sub(r"^(the|a|an)\s+", "", noun, flags=re.IGNORECASE)
    
    # 5. Resolve Category
    category = matcher.get_category(noun)
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