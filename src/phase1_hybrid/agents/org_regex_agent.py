import re
from pathlib import Path
from typing import List, Dict, Any

class OrgRegexAgent:
    def __init__(self, regex_path=None):
        if regex_path is None:
            regex_path = Path("data/knowledge_base/ORG/patterns.txt")
        
        self.regex_path = regex_path
        self.patterns = self._load_patterns()
        
        # Εσωτερική Blacklist για το Regex (για τα κεφαλαία ακρωνύμια)
        self.blacklist = {
            "ΚΑΙ", "ΤΟΥ", "ΤΗΝ", "ΤΩΝ", "ΤΟΥΣ", "ΤΙΣ", "ΜΙΑ", "ΕΝΑ", "ΑΠΟ", 
            "ΓΙΑ", "ΜΕ", "ΣΕ", "ΩΣ", "ΔΙΑ", "ΕΠΙ", "ΥΠΟ", "ΠΡΟΣ", "ΑΠΟΦΑΣΗ",
            "ΝΟΜΟΣ", "ΑΡΘΡΟ", "ΦΕΚ", "ΕΤΟΣ", "ΕΥΡΩ", "ΔΡΧ",
            "ΠΡΟΕΔΡΙΚΟ", "ΔΙΑΤΑΓΜΑ", "ΑΡΙΘΜ", "ΔΗΜΟΚΡΑΤΙΑΣ", "ΕΛΛΗΝΙΚΗΣ", 
            "ΠΡΟΕΔΡΟΣ", "ΤΗΣ", "ΑΝΑΠΛΗΡΩΤΗΣ", "ΥΠΟΥΡΓΟΣ", "ΓΕΝΙΚΟΣ", "ΕΙΔΙΚΟΣ",
            "ΘΕΜΑ", "ΣΧΕΤ", "ΑΠΟΦΑΣΙΖΟΥΜΕ", "ΕΧΟΝΤΑΣ", "ΥΠΟΨΗ", "ΟΙ", "ΤΑ", "ΤΟ",
            "ΟΠΩΣ", "ΙΣΧΥΕΙ", "ΠΑΡ", "ΠΕΡ", "ΕΔΑΦ", "ΤΕΥΧΟΣ",
            "ΜΗΤΣΟΤΑΚΗΣ", "ΕΕΠ"
        }

    def _load_patterns(self):
        patterns = []
        if isinstance(self.regex_path, Path) and self.regex_path.exists():
            try:
                with open(self.regex_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                # Removed re.IGNORECASE to prevent [Α-Ω] from matching lowercase
                                patterns.append(re.compile(line))
                            except re.error as e:
                                print(f"⚠️ Invalid ORG Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading ORG patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        results = []
        used_positions = set()
        
        # Σημαντικό: Ταξινόμηση patterns; Όχι απαραίτητα, θα κάνουμε resolution μετά.
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                # Capturing Group Logic
                if match.groups():
                    target_group_index = next((i for i, g in enumerate(match.groups()) if g is not None), None)
                    if target_group_index is not None:
                        match_text = match.group(target_group_index + 1)
                        if not match_text: continue
                        start = match.start(target_group_index + 1)
                        end = match.end(target_group_index + 1)
                    else:
                        match_text = match.group()
                        start = match.start()
                        end = match.end()
                else:
                    match_text = match.group()
                    start = match.start()
                    end = match.end()

                # 1. Φίλτρο Μήκους
                if len(match_text) < 3: continue
                
                # 2. Φίλτρο Blacklist (για τα κεφαλαία)
                if match_text.upper() in self.blacklist: continue

                # 3. Φίλτρο Περιεχομένου: Αν έχει μόνο αριθμούς ή σύμβολα
                if not any(c.isalpha() for c in match_text): continue
                
                # Check Overlap
                if any(pos in used_positions for pos in range(start, end)):
                    continue
                
                for pos in range(start, end):
                    used_positions.add(pos)
                
                # Confidence Logic: Αν ξεκινάει με "Υπουργείο", είναι 100% σωστό
                conf = 0.90
                if match_text.lower().startswith(('υπουργ', 'γενικ', 'συμβούλ')):
                    conf = 0.98

                results.append({
                    "text": match_text,
                    "label": "ORG",
                    "start": start,
                    "end": end,
                    "confidence": conf,
                    "source": "Regex"
                })
        
        # Resolution: Longest Match Wins (π.χ. "Υπουργείο Οικονομικών" vs "Οικονομικών")
        results.sort(key=lambda x: len(x['text']), reverse=True)
        
        # (Θα μπορούσαμε να ξανακάνουμε check overlap εδώ για τα sorted results, 
        # αλλά το κάναμε ήδη greedy παραπάνω. Για απόλυτη ακρίβεια, το greedy loop 
        # πρέπει να γίνεται ΑΦΟΥ μαζέψουμε όλα τα matches.)
        
        # ΑΣ ΤΟ ΚΑΝΟΥΜΕ ΣΩΣΤΑ: Re-filtering
        final_results = []
        final_mask = [False] * len(text)
        
        for res in results:
            is_free = True
            for i in range(res['start'], res['end']):
                if final_mask[i]:
                    is_free = False
                    break
            
            if is_free:
                final_results.append(res)
                for i in range(res['start'], res['end']):
                    final_mask[i] = True
                    
        return final_results