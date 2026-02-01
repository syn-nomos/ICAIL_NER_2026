import re
from pathlib import Path
from typing import List, Dict, Any

class DateRegexAgent:
    def __init__(self, regex_path=None):
        if regex_path is None:
            regex_path = Path("data/knowledge_base/DATE/patterns.txt")
        
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
                                # IGNORECASE είναι βασικό για μήνες
                                patterns.append(re.compile(line, re.IGNORECASE))
                            except re.error as e:
                                print(f"⚠️ Invalid Date Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading DATE patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        raw_matches = []
        
        # 1. Εύρεση όλων των matches
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                matched_text = match.group()
                
                # Validation: Αποφεύγουμε σκέτους αριθμούς που δεν είναι έτη
                # Αν είναι μόνο 4 ψηφία (π.χ. 2500), πρέπει να είμαστε σίγουροι
                if matched_text.isdigit() and len(matched_text) == 4:
                    year = int(matched_text)
                    if year < 1900 or year > 2100:
                        continue # Πιθανώς ποσό ή κωδικός, όχι ημερομηνία
                
                raw_matches.append({
                    "text": matched_text,
                    "label": "DATE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.98, # Πολύ υψηλό confidence για Regex Dates
                    "source": "Regex"
                })
        
        if not raw_matches:
            return []

        # 2. RESOLUTION: Longest Match Wins (Για να μην πιάνει το "2009" μέσα στο "18-2-2009")
        raw_matches.sort(key=lambda x: len(x['text']), reverse=True)
        
        final_matches = []
        occupied_mask = [False] * len(text)
        
        for match in raw_matches:
            start, end = match['start'], match['end']
            
            # Έλεγχος επικάλυψης
            is_free = True
            for i in range(start, end):
                if occupied_mask[i]:
                    is_free = False
                    break
            
            if is_free:
                final_matches.append(match)
                for i in range(start, end):
                    occupied_mask[i] = True
                    
        # Ταξινόμηση με βάση τη θέση στο κείμενο
        final_matches.sort(key=lambda x: x['start'])
        
        return final_matches