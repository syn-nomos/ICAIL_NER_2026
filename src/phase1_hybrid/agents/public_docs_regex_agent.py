import re
from pathlib import Path
from typing import List, Dict, Any

class PublicDocsRegexAgent:
    def __init__(self, regex_path=None):
        if regex_path is None:
            regex_path = Path("data/knowledge_base/PUBLIC_DOCS/patterns.txt")
        
        self.regex_path = regex_path
        self.patterns = self._load_patterns()

    def _load_patterns(self):
        patterns = []
        if isinstance(self.regex_path, Path) and self.regex_path.exists():
            try:
                with open(self.regex_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                # UNICODE + IGNORECASE
                                patterns.append(re.compile(line, re.IGNORECASE | re.UNICODE))
                            except re.error as e:
                                print(f"⚠️ Invalid PUBLIC-DOCS Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading PUBLIC-DOCS patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        raw_matches = []
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                match_text = match.group()
                
                # Φίλτρο: Να μην πιάνει σκέτα νούμερα μικρά (π.χ. "2000")
                if match_text.isdigit() and len(match_text) < 5:
                    continue
                
                # --- RULE: IGNORE IF INSIDE PARENTHESES ---
                # Check surrounding context for (...)
                # Find closest non-space char before
                pre_char = ''
                idx = match.start() - 1
                while idx >= 0:
                    if text[idx] != ' ':
                        pre_char = text[idx]
                        break
                    idx -= 1
                
                # Find closest non-space char after
                post_char = ''
                idx = match.end()
                while idx < len(text):
                    if text[idx] != ' ':
                        post_char = text[idx]
                        break
                    idx += 1
                
                # If enclosed in parens, assume it's metadata (citation) and IGNORE
                if pre_char == '(' and post_char == ')':
                    continue
                    
                raw_matches.append({
                    "text": match_text,
                    "label": "PUBLIC-DOCS",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.95,
                    "source": "Rule: Regex (High Precision)"
                })
        
        if not raw_matches:
            return []

        # Resolution: Longest Match Wins
        raw_matches.sort(key=lambda x: len(x['text']), reverse=True)
        
        final_matches = []
        occupied_mask = [False] * len(text)
        
        for match in raw_matches:
            start, end = match['start'], match['end']
            
            is_free = True
            for i in range(start, end):
                if occupied_mask[i]:
                    is_free = False
                    break
            
            if is_free:
                final_matches.append(match)
                for i in range(start, end):
                    occupied_mask[i] = True
                    
        return final_matches