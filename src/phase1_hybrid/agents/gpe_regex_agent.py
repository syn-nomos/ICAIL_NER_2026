import re
from pathlib import Path
from typing import List, Dict, Any

class GpeRegexAgent:
    def __init__(self, regex_path=None):
        if regex_path is None:
            regex_path = Path("data/knowledge_base/GPE/patterns.txt")
        
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
                                # REMOVED re.IGNORECASE to enforce capitalization rules in patterns
                                patterns.append(re.compile(line))
                            except re.error as e:
                                print(f"⚠️ Invalid GPE Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading GPE patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        raw_matches = []
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                match_text = match.group()
                
                # Φίλτρο: Αν έπιασε μόνο "Δήμος " χωρίς όνομα, το αγνοούμε
                if len(match_text.strip()) <= 5:
                    continue

                raw_matches.append({
                    "text": match_text,
                    "label": "GPE",
                    "start": match.start(),
                    "end": match.end(),
                    # Τα Regex με "Δήμος/Νομός" είναι σχεδόν πάντα σωστά
                    "confidence": 0.96, 
                    "source": "Regex"
                })
        
        if not raw_matches:
            return []

        # Resolution: Longest Match Wins
        # Π.χ. Αν πιάσει "Δήμου Κεφαλονιάς" (Long) και "Κεφαλονιάς" (Short), κρατάμε το Long.
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