import re
import json
from pathlib import Path
from typing import List, Dict, Any

class LegRefsLexiconAgent:
    def __init__(self, lexicon_path=None):
        if lexicon_path is None:
            # Try json first
            json_path = Path("data/knowledge_base/LEG_REFS/lexicon.json")
            if json_path.exists():
                lexicon_path = json_path
            else:
                lexicon_path = Path("data/knowledge_base/LEG_REFS/lexicon.txt")
        
        self.lexicon_path = lexicon_path
        self.lexicon = []
        self._load_lexicon()

    def _load_lexicon(self):
        if self.lexicon_path.exists():
            try:
                if self.lexicon_path.suffix == '.json':
                    with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.lexicon = [str(x).strip() for x in data if len(str(x)) >= 3]
                else:
                    with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            term = line.strip()
                            if term and len(term) >= 3:
                                self.lexicon.append(term)
            except Exception as e:
                print(f"Error loading LEG_REFS lexicon: {e}")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        
        # Ταξινόμηση Longest Match First
        sorted_terms = sorted(self.lexicon, key=len, reverse=True)
        used_positions = set()
        
        for term in sorted_terms:
            term_lower = term.lower()
            start_pos = 0
            while True:
                idx = text_lower.find(term_lower, start_pos)
                if idx == -1:
                    break
                
                end_idx = idx + len(term)
                
                # Word Boundary Check (για να μην πιάνει μέρος λέξης)
                is_whole_word = True
                if idx > 0 and text[idx-1].isalnum(): is_whole_word = False
                if end_idx < len(text) and text[end_idx].isalnum(): is_whole_word = False
                
                if is_whole_word:
                    # Check Overlap
                    if not any(pos in used_positions for pos in range(idx, end_idx)):
                        for pos in range(idx, end_idx):
                            used_positions.add(pos)
                        
                        results.append({
                            "text": text[idx:end_idx],
                            "label": "LEG-REFS",
                            "start": idx,
                            "end": end_idx,
                            "confidence": 1.0,
                            "source": "Lexicon"
                        })
                
                start_pos = idx + 1
                
        return results