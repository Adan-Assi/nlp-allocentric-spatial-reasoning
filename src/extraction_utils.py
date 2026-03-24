import spacy
import config

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
        
        text_lower = text.lower()
        
        # 1. Check if the word is a synonym in our Map (e.g., "deli" -> "SHOP")
        if text_lower in self.text_lookup:
            return self.text_lookup[text_lower]
        
        # 2. Check if the word is the Group Name itself (e.g., "pharmacy" -> "PHARMACY")
        if text_lower in self.group_names:
            return self.group_names[text_lower]
        
        # 3. Partial Match for compound brands (e.g., "Starbucks" contains "cafe" logic)
        # We check if any of our trigger words exist inside the extracted text
        for trigger, group in self.text_lookup.items():
            if trigger in text_lower:
                return group
                
        return "UNKNOWN"

# Instantiate the matcher
matcher = CategoricalMatcher()

def extract_rvs_target(text: str) -> str:
    """
    Extracts the landmark name and resolves it to a Category.
    """
    doc = nlp(text)
    potential_target = None

    # 1. Syntactic Extraction: Find the object of the spatial preposition
    # This identifies "Deli" as the target in "Meet at the Deli"
    for token in doc:
        if token.text.lower() in ['at', 'to', 'near', 'by', 'past', 'on']:
            for child in token.children:
                if child.pos_ in ["NOUN", "PROPN"]:
                    clean_word = child.text.lower()
                    # Filter out generic street terms to avoid snapping to the road instead of the shop
                    if clean_word not in ["street", "avenue", "road", "block", "corner", "north", "south", "east", "west"]:
                        potential_target = clean_word
                        break
            if potential_target: break
    
    # 2. Fallback: If no prepositional object, look for any noun that matches a known category
    if not potential_target:
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                cat = matcher.get_category(token.text)
                if cat != "UNKNOWN":
                    return cat

    # 3. Final Resolution: Map the name to a Category or return the raw name
    if potential_target:
        category = matcher.get_category(potential_target)
        return category if category != "UNKNOWN" else potential_target

    return "unknown"