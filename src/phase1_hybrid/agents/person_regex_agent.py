import re
import json
from pathlib import Path
from typing import List, Dict, Any

class PersonRegexAgent:
    def __init__(self, regex_path=None, lexicon_path=None):
        if regex_path is None:
            regex_path = Path("data/knowledge_base/PERSON/patterns.txt")
        if lexicon_path is None:
            lexicon_path = Path("data/knowledge_base/PERSON/lexicon.json")
        
        self.regex_path = regex_path
        self.lexicon_path = lexicon_path
        self.patterns = self._load_patterns()
        self.known_first_names = self._load_first_names()
        
        # Λέξεις που μοιάζουν με ονόματα (Κεφαλαία) αλλά ΔΕΝ είναι
        self.blacklist = {
            "ΝΟΜΟΣ", "ΑΡΘΡΟ", "ΑΠΟΦΑΣΗ", "ΔΙΑΤΑΓΜΑ", "ΕΓΚΥΚΛΙΟΣ", "ΥΠΟΥΡΓΕΙΟ", 
            "ΔΗΜΟΣ", "ΝΟΜΟΣ", "ΣΩΜΑ", "ΤΜΗΜΑ", "ΓΡΑΦΕΙΟ", "ΣΥΜΒΟΥΛΙΟ", 
            "ΕΠΙΤΡΟΠΗ", "ΔΙΕΥΘΥΝΣΗ", "ΕΛΛΑΔΑ", "ΑΘΗΝΑ", "ΕΤΑΙΡΕΙΑ", "ΟΜΙΛΟΣ",
            "ΤΗΣ", "ΤΟΥ", "ΤΩΝ", "ΤΟΝ", "ΤΗΝ", "ΚΑΙ", "ΓΙΑ", "ΜΕ", "ΣΕ", "ΑΠΟ",
            "ΟΙ", "ΤΑ", "ΤΟ", "ΣΤΟΥΣ", "ΣΤΙΣ", "ΣΤΑ", "ΩΣ", "ΠΡΟΣ", "ΚΑΤΑ", "ΔΙΑ",
            "ΥΠΟΥΡΓΟΣ", "ΥΦΥΠΟΥΡΓΟΣ", "ΑΝΑΠΛΗΡΩΤΗΣ", "ΓΕΝΙΚΟΣ", "ΕΙΔΙΚΟΣ", "ΓΡΑΜΜΑΤΕΑΣ",
            "ΠΡΟΕΔΡΟΣ", "ΔΙΟΙΚΗΤΗΣ", "ΔΙΕΥΘΥΝΤΗΣ", "ΠΡΟΙΣΤΑΜΕΝΟΣ", "ΒΟΥΛΕΥΤΗΣ",
            "ΕΛΛΗΝΙΚΗΣ", "ΔΗΜΟΚΡΑΤΙΑΣ", "ΚΥΒΕΡΝΗΣΗΣ", "ΕΦΗΜΕΡΙΣ", "ΤΕΥΧΟΣ",
            "ΟΙΚΟΝΟΜΙΚΩΝ", "ΕΣΩΤΕΡΙΚΩΝ", "ΑΝΑΠΤΥΞΗΣ", "ΠΑΙΔΕΙΑΣ", "ΥΓΕΙΑΣ",
            "ΔΙΚΑΙΟΣΥΝΗΣ", "ΕΘΝΙΚΗΣ", "ΑΜΥΝΑΣ", "ΕΞΩΤΕΡΙΚΩΝ", "ΠΟΛΙΤΙΣΜΟΥ",
            "ΠΕΡΙΒΑΛΛΟΝΤΟΣ", "ΕΝΕΡΓΕΙΑΣ", "ΕΡΓΑΣΙΑΣ", "ΚΟΙΝΩΝΙΚΗΣ", "ΑΣΦΑΛΙΣΗΣ",
            "ΑΓΡΟΤΙΚΗΣ", "ΤΟΥΡΙΣΜΟΥ", "ΝΑΥΤΙΛΙΑΣ", "ΜΕΤΑΝΑΣΤΕΥΣΗΣ", "ΨΗΦΙΑΚΗΣ",
            "ΚΛΙΜΑΤΙΚΗΣ", "ΚΡΙΣΗΣ", "ΠΡΟΣΤΑΣΙΑΣ", "ΠΟΛΙΤΗ", "ΔΗΜΟΣΙΑΣ", "ΤΑΞΗΣ",
            "ΜΑΘΗΜΑ", "ΜΑΘΗΜΑΤΑ", "ΕΞΑΜΗΝΟ", "ΘΕΩΡΙΑ", "ΣΥΣΤΗΜΑ", "ΣΥΣΤΗΜΑΤΑ",
            "ΣΥΣΤΗΜΑΤΩΝ", "ΑΝΑΛΥΣΗ", "ΑΝΑΛΥΣΗΣ", "ΣΧΕΔΙΑΣΜΟΣ", "ΣΧΕΔΙΑΣΗ",
            "ΕΡΕΥΝΑ", "ΕΡΕΥΝΑΣ", "ΚΕΦΑΛΑΙΟ", "ΚΕΦΑΛΑΙΑ", "ΕΠΙΛΟΓΗ", "ΕΠΙΛΟΓΗΣ",
            "ΚΟΡΜΟΥ", "ΕΙΔΙΚΑ", "ΜΟΝΟ", "ΕΝΟΣ", "ΣΤΟΧΑΣΤΙΚΑ", "ΚΑΤΑΝΕΜΗΜΕΝΑ",
            "ΠΡΟΗΓΜΕΝΑ", "ΜΕΘΟΔΟΙ", "ΤΕΧΝΟΛΟΓΙΚΟΣ", "ΣΤΡΑΤΗΓΙΚΗΣ", "ΜΑΧΗΣ",
            "ΔΙΑΚΡΙΤΩΝ", "ΕΛΕΥΘΕΡΗΣ", "ΥΠΟΧΡΕΩΤΙΚΑ", "ΚΑΤΕΥΘΥΝΣΗΣ",
            "ΔΙΑΧΕΙΡΙΣΗ", "ΑΞΙΟΠΙΣΤΙΑΣ", "ΟΙΚΟΝΟΜΙΚΗ", "ΠΡΟΣΕΓΓΙΣΗ", "ΕΝΝΟΙΕΣ",
            "ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΥ", "ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΣ", "ΕΠΑΝΔΡΩΜΕΝΑ", "ΡΟΜΠΟΤΙΚΑ",
            "ΕΚΤΙΜΗΣΗ", "ΚΟΣΤΟΥΣ", "ΕΙΣΑΓΩΓΗ", "ΠΡΟΧΩΡΗΜΕΝΑ", "ΘΕΜΑΤΑ",
            "ΚΑΤΑΝΕΜΗΜΕΝΗ", "ΤΕΧΝΗΤΗ", "ΠΟΛΛΑΠΛΩΝ", "ΠΑΡΑΓΟΝΤΩΝ", "ΜΕΤΑΠΤΥΧΙΑΚΗ",
            "ΔΙΠΛΩΜΑΤΙΚΗ", "ΜΟΝΤΕΛΟΠΟΙΗΣΗ", "ΚΙΝΔΥΝΟΥ", "ΜΑΘΗΜΑΤΙΚΟΣ",
            "ΣΤΡΑΤΙΩΤΙΚΩΝ", "ΑΠΟΦΑΣΕΩΝ", "ΕΠΙΧΕΙΡΗΣΙΑΚΟΙ", "ΑΛΓΟΡΙΘΜΟΙ",
            "ΣΤΑΤΙΣΤΙΚΟΥ", "ΕΛΕΓΧΟΥ", "ΠΟΙΟΤΗΤΑΣ", "ΜΟΝΤΕΛΑ", "ΠΡΟΒΛΕΨΗΣ",
            "ΑΥΤΟΜΑΤΟΣ", "ΕΛΕΓΧΟΣ", "ΕΡΓΑΣΙΑ", "ΕΡΓΑΣΙΑΣ", "ΔΙΑΛΕΞΕΙΣ",
            "ΑΣΚΗΣΕΙΣ", "ΕΡΓΑΣΤΗΡΙΟ", "ΕΡΓΑΣΤΗΡΙΑ", "ΘΕΩΡΗΤΙΚΟ", "ΜΕΡΟΣ",
            "ΕΡΓΩΝ", "ΟΙΚΟΝΟΜΙΑΣ", "ΤΑΞΗΣ", "ΔΙΑΦΟΡΑ", "ΠΛΟΙΑ", "ΠΡΟΤΕΙΝΕΤΑΙ", "ΑΡΣΗ",
            "ΘΕΜΑΤΑ", "ΣΧΟΛΙΑ", "ΠΑΡΑΤΗΡΗΣΕΙΣ", "ΚΡΙΤΗΡΙΑ", "ΣΥΜΠΕΡΑΣΜΑΤΑ", "ΕΙΔΗ",
            "ΚΑΤΗΓΟΡΙΕΣ", "ΜΕΤΡΑ", "ΟΡΟΙ", "ΠΡΟΥΠΟΘΕΣΕΙΣ", "ΣΧΕΔΙΟ", "ΜΕΛΕΤΗ"
        }

    def _load_first_names(self):
        names = set()
        if self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "first_names" in data:
                        for name in data["first_names"]:
                            names.add(name.upper())
            except Exception as e:
                print(f"❌ Error loading PERSON lexicon for regex: {e}")
        return names

    def _load_patterns(self):
        patterns = []
        if isinstance(self.regex_path, Path) and self.regex_path.exists():
            try:
                with open(self.regex_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                patterns.append(re.compile(line))
                            except re.error as e:
                                print(f"⚠️ Invalid PERSON Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading PERSON patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        results = []
        used_positions = set()
        
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
                
                # 1. Blacklist Check (Κρίσιμο για Κεφαλαία)
                words = match_text.split()
                if any(w.upper() in self.blacklist for w in words):
                    continue
                    
                # 2. Length Check
                if len(match_text) < 5: continue

                # 3. Common Name Check for Uppercase Matches
                # If the match is strictly uppercase and consists of 2 words,
                # we require at least one word to be a known first name.
                # This filters out "COURSE TITLE" garbage.
                if match_text.isupper():
                    words = match_text.split()
                    if len(words) == 2:
                        # Check if either word is a known first name
                        w1 = words[0]
                        w2 = words[1]
                        
                        # If we have a list of known names, use it.
                        if self.known_first_names:
                            is_w1_name = w1 in self.known_first_names
                            is_w2_name = w2 in self.known_first_names
                            
                            # If neither is a known name, treat as garbage
                            if not (is_w1_name or is_w2_name):
                                continue

                start = match.start()
                end = match.end()
                
                if any(pos in used_positions for pos in range(start, end)):
                    continue
                
                for pos in range(start, end):
                    used_positions.add(pos)
                
                # Confidence: Αν έχει τελεία (π.χ. ΓΡ .), είναι πολύ πιθανό να είναι όνομα
                conf = 0.90
                if '.' in match_text:
                    conf = 0.96

                results.append({
                    "text": match_text,
                    "label": "PERSON",
                    "start": start,
                    "end": end,
                    "confidence": conf,
                    "source": "Regex"
                })
        
        return results