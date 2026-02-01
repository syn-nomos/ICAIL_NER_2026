import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

class FacilityLexiconAgent:
    def __init__(self, lexicon_path=None):
        if lexicon_path is None:
            lexicon_path = Path("data/knowledge_base/FACILITY") / "lexicon.json" # ή facility_lexicon.json
        self.lexicon_path = lexicon_path
        self.lexicon = []
        self.load_lexicon()

    def load_lexicon(self):
        if self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.lexicon = data
                    elif isinstance(data, dict):
                        # Αν το json έχει δομή {"facility_terms": [...]}, προσάρμοσέ το εδώ
                        self.lexicon = data.get('facility_terms', [])
            except Exception as e:
                print(f"Error loading lexicon: {e}")

    def predict(self, text: str, entity_type: str = "FACILITY") -> List[Dict[str, Any]]:
        results = []
        text_normalized = remove_accents(text.lower())
        
        # Ταξινόμηση βάσει μήκους (Longest Match First)
        # Προστασία: Βεβαιωνόμαστε ότι το term είναι string
        sorted_terms = sorted([str(t) for t in self.lexicon], key=len, reverse=True)
        
        used_positions = set()
        
        for term in sorted_terms:
            # Case Sensitivity Logic
            # If term is all uppercase and short (< 5 chars), enforce case match
            is_acronym = term.isupper() and len(term) < 5
            
            if is_acronym:
                # Search in original text (Exact Match)
                search_text = text
                search_term = term
            else:
                # Case insensitive & Accent insensitive
                search_text = text_normalized
                search_term = remove_accents(term.lower().strip())

            if len(search_term) < 2: continue
            
            start_pos = 0
            while True:
                idx = search_text.find(search_term, start_pos)
                if idx == -1: break
                
                end_idx = idx + len(search_term)
                
                # Word Boundary Check
                # We check boundaries on the ORIGINAL text to be safe
                is_whole_word = True
                if idx > 0 and text[idx-1].isalnum(): is_whole_word = False
                if end_idx < len(text) and text[end_idx].isalnum(): is_whole_word = False
                
                if is_whole_word:
                    if not any(pos in used_positions for pos in range(idx, end_idx)):
                        for pos in range(idx, end_idx):
                            used_positions.add(pos)
                        
                        results.append({
                            "text": text[idx:end_idx], # Return original text span
                            "label": entity_type,
                            "start": idx,
                            "end": end_idx,
                            "confidence": 1.0,
                            "source": "Lexicon"
                        })
                
                start_pos = idx + 1
                
        return results