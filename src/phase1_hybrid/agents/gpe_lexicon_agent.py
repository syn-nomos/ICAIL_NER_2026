import json
from pathlib import Path
from typing import List, Dict, Any

class GpeLexiconAgent:
    def __init__(self, lexicon_path=None):
        if lexicon_path is None:
            lexicon_path = Path("data/knowledge_base/GPE/lexicon.json")
        self.lexicon_path = lexicon_path
        self.lexicon = []
        self._load_lexicon()

    def _load_lexicon(self):
        if self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Υποστήριξη και για απλή λίστα και για dict
                    if isinstance(data, list):
                        self.lexicon = data
                    elif isinstance(data, dict):
                        # Αν είναι dict, παίρνουμε όλα τα values (flatten)
                        self.lexicon = []
                        for key, val in data.items():
                            if isinstance(val, list): self.lexicon.extend(val)
            except Exception as e:
                print(f"Error loading GPE lexicon: {e}")

    def predict(self, text: str, entity_type: str = "GPE") -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        
        # Ταξινόμηση: Μεγαλύτερα πρώτα
        sorted_terms = sorted([str(t) for t in self.lexicon], key=len, reverse=True)
        used_positions = set()
        
        for term in sorted_terms:
            term = term.strip()
            if len(term) < 2: continue # Αγνοούμε πολύ μικρά (π.χ. "Α")
            
            term_lower = term.lower()
            start_pos = 0
            
            while True:
                idx = text_lower.find(term_lower, start_pos)
                if idx == -1: break
                
                end_idx = idx + len(term)
                
                # --- WORD BOUNDARY CHECK ---
                # Ελέγχουμε αν είναι μέρος άλλης λέξης (π.χ. "Χίος" στο "ΣτοιΧείος")
                is_whole_word = True
                
                # Check Left
                if idx > 0 and text[idx-1].isalnum(): is_whole_word = False
                # Check Right
                if end_idx < len(text) and text[end_idx].isalnum(): is_whole_word = False
                
                if is_whole_word:
                    # Check Overlap
                    if not any(pos in used_positions for pos in range(idx, end_idx)):
                        for pos in range(idx, end_idx):
                            used_positions.add(pos)
                        
                        results.append({
                            "text": text[idx:end_idx],
                            "label": "GPE",
                            "start": idx,
                            "end": end_idx,
                            "confidence": 1.0, 
                            "source": "Lexicon"
                        })
                
                start_pos = idx + 1
                
        return results