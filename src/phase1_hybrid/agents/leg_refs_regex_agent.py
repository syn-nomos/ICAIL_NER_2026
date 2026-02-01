import re
from pathlib import Path
from typing import List, Dict, Any

class LegRefsRegexAgent:
    def __init__(self, regex_path=None):
        if regex_path is None:
            # Υποθέτουμε ότι τρέχει από το root
            regex_path = Path("data/knowledge_base/LEG_REFS/patterns.txt")
        
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
                                # Check for case sensitive marker (?c)
                                flags = re.IGNORECASE | re.DOTALL
                                if line.startswith('(?c)'):
                                    line = line[4:]
                                    flags = re.DOTALL # No IGNORECASE
                                
                                patterns.append(re.compile(line, flags))
                            except re.error as e:
                                print(f"⚠️ Invalid Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        raw_matches = []
        
        # 1. Βρες ΟΛΑ τα πιθανά matches
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                raw_matches.append({
                    "text": match.group(),
                    "label": "LEG-REFS",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                    "source": "Regex"
                })
        
        if not raw_matches:
            return []

        # 2. RESOLUTION STRATEGY: Longest Match Wins (Κερδίζει το μεγαλύτερο)
        # Αν έχουμε "Ν. 1234" (len 7) και "Ν. 1234 «Τίτλος»" (len 20)
        # Θέλουμε το δεύτερο.
        
        # Ταξινομούμε: Πρώτα κατά μήκος (φθίνουσα), μετά κατά θέση
        raw_matches.sort(key=lambda x: len(x['text']), reverse=True)
        
        final_matches = []
        occupied_mask = [False] * len(text)
        
        for match in raw_matches:
            start, end = match['start'], match['end']
            
            # Έλεγχος: Είναι ελεύθερος ο χώρος;
            is_free = True
            for i in range(start, end):
                if occupied_mask[i]:
                    is_free = False
                    break
            
            if is_free:
                final_matches.append(match)
                # Μαρκάρουμε τον χώρο
                for i in range(start, end):
                    occupied_mask[i] = True
                    
        # Τελική ταξινόμηση βάσει θέσης στο κείμενο (για ομορφιά)
        final_matches.sort(key=lambda x: x['start'])
        
        return final_matches