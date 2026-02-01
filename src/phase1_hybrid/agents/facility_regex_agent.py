import re
from pathlib import Path
from typing import List, Dict, Any

class FacilityRegexAgent:
    def __init__(self, regex_path=None):
        if regex_path is None:
            regex_path = Path("data/knowledge_base/FACILITY/patterns.txt")
        
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
                                # REMOVED re.IGNORECASE to enforce capitalization rules
                                patterns.append(re.compile(line, re.UNICODE))
                            except re.error as e:
                                print(f"⚠️ Invalid Facility Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading FACILITY patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        raw_matches = []
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                text_span = match.group()
                
                # Φίλτρο Ποιότητας:
                # Αν έπιασε μόνο "Οδός" (χωρίς όνομα), το πετάμε.
                # Πρέπει να έχει μήκος > 5 και τουλάχιστον ένα κενό (ένδειξη ότι έχει όνομα)
                if len(text_span) < 5 or ' ' not in text_span:
                    continue

                raw_matches.append({
                    "text": text_span,
                    "label": "FACILITY",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.90, # Λίγο χαμηλότερο από LEG-REFS γιατί τα κτίρια έχουν παραλλαγές
                    "source": "Regex"
                })
        
        if not raw_matches:
            return []

        # Resolution: Longest Match Wins
        # Αν πιάσει "Ε.Ο. Αθηνών" και "Ε.Ο. Αθηνών - Λαμίας", θέλουμε το δεύτερο.
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