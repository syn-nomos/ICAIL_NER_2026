import json
from pathlib import Path
from typing import List, Dict, Any

class PublicDocsLexiconAgent:
    def __init__(self, lexicon_path=None):
        if lexicon_path is None:
            lexicon_path = Path("data/knowledge_base/PUBLIC_DOCS/lexicon.json")
        self.lexicon_path = lexicon_path
        self.lexicon = []
        self._load_lexicon()

    def _load_lexicon(self):
        if self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.lexicon = data
            except Exception as e:
                print(f"Error loading PUBLIC-DOCS lexicon: {e}")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        
        sorted_terms = sorted([str(t) for t in self.lexicon], key=len, reverse=True)
        used_positions = set()
        
        for term in sorted_terms:
            term_lower = term.lower().strip()
            if len(term_lower) < 3: continue
            
            start_pos = 0
            while True:
                idx = text_lower.find(term_lower, start_pos)
                if idx == -1: break
                
                end_idx = idx + len(term_lower)
                
                # Word Boundary
                is_whole_word = True
                if idx > 0 and text[idx-1].isalnum(): is_whole_word = False
                if end_idx < len(text) and text[end_idx].isalnum(): is_whole_word = False
                
                if is_whole_word:
                    if not any(pos in used_positions for pos in range(idx, end_idx)):
                        for pos in range(idx, end_idx): used_positions.add(pos)
                        
                        results.append({
                            "text": text[idx:end_idx],
                            "label": "PUBLIC_DOCS",
                            "start": idx,
                            "end": end_idx,
                            "confidence": 1.0,
                            "source": "Lexicon"
                        })
                
                start_pos = idx + 1
                
        return results