import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

class PersonLexiconAgent:
    def __init__(self, lexicon_path=None):
        if lexicon_path is None:
            lexicon_path = Path("data/knowledge_base/PERSON/lexicon.json")
        
        self.lexicon_path = lexicon_path
        self.first_names = set()
        self.surnames = set()
        self.full_names = set() # VIPs (compounds)
        self._load_lexicon()

    def _load_lexicon(self):
        if self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict):
                        # Lowercase AND remove accents
                        self.first_names = {remove_accents(x.lower()) for x in data.get('first_names', [])}
                        self.surnames = {remove_accents(x.lower()) for x in data.get('surnames', [])}
                        self.full_names = {remove_accents(x.lower()) for x in data.get('compounds', [])}
                    elif isinstance(data, list):
                        self.full_names = {remove_accents(x.lower()) for x in data}
                        
            except Exception as e:
                print(f"Error loading PERSON lexicon: {e}")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        results = []
        # Χωρίζουμε το κείμενο σε λέξεις κρατώντας τα offsets
        words = []
        for match in re.finditer(r'\b\w+\b', text):
            words.append(match)
            
        used_positions = set()
        
        # Normalize text for matching
        text_normalized = remove_accents(text.lower())
        
        # 1. VIP CHECK (Full Names Exact Match)
        for vip in self.full_names:
            start_pos = 0
            while True:
                idx = text_normalized.find(vip, start_pos)
                if idx == -1: break
                end_idx = idx + len(vip)
                
                # Overlap check
                if not any(pos in used_positions for pos in range(idx, end_idx)):
                     for pos in range(idx, end_idx): used_positions.add(pos)
                     results.append({
                        "text": text[idx:end_idx], # Return original text
                        "label": "PERSON",
                        "start": idx,
                        "end": end_idx,
                        "confidence": 1.0,
                        "source": "Lexicon_VIP"
                     })
                start_pos = idx + 1

        # 2. COMBINATORIAL CHECK (Όνομα + Επίθετο)
        for i in range(len(words) - 1):
            w1 = words[i].group()
            w2 = words[i+1].group()
            
            if words[i+1].start() - words[i].end() > 4: 
                continue

            # Normalize words
            w1_norm = remove_accents(w1.lower())
            w2_norm = remove_accents(w2.lower())

            is_fname = w1_norm in self.first_names
            is_sname = w2_norm in self.surnames
            
            if (is_fname and (is_sname or w2_norm in self.first_names)):
                start = words[i].start()
                end = words[i+1].end()
                
                if not any(pos in used_positions for pos in range(start, end)):
                    for pos in range(start, end): used_positions.add(pos)
                    results.append({
                        "text": text[start:end],
                        "label": "PERSON",
                        "start": start,
                        "end": end,
                        "confidence": 0.95,
                        "source": "Lexicon_Combo"
                    })
        
        # 3. STANDALONE SURNAME CHECK (Experimental)
        # Only if Capitalized and in Surnames list
        for match in words:
            w = match.group()
            if len(w) < 4: continue
            
            # Must be Capitalized or Uppercase
            if not (w[0].isupper()): continue
            
            w_norm = remove_accents(w.lower())
            if w_norm in self.surnames:
                start = match.start()
                end = match.end()
                
                if not any(pos in used_positions for pos in range(start, end)):
                    for pos in range(start, end): used_positions.add(pos)
                    results.append({
                        "text": text[start:end],
                        "label": "PERSON",
                        "start": start,
                        "end": end,
                        "confidence": 0.85,
                        "source": "Lexicon_Surname"
                    })

        return results