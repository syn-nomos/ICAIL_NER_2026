import json
import re
import os

class KnowledgeBaseAgent:
    def __init__(self, entity_name, kb_folder="data/knowledge_base"):
        self.entity_name = entity_name
        self.patterns = []
        self.lexicon_phrases = set()
        
        # Paths
        entity_path = os.path.join(kb_folder, entity_name)
        regex_path = os.path.join(entity_path, "patterns.txt")
        lexicon_path = os.path.join(entity_path, "lexicon.json")

        # 1. Φόρτωση Regex (αν υπάρχει το αρχείο)
        if os.path.exists(regex_path):
            self._load_regex(regex_path)
            
        # 2. Φόρτωση Lexicon (αν υπάρχει)
        if os.path.exists(lexicon_path):
            self._load_lexicon(lexicon_path)

    def _load_regex(self, path):
        print(f"[{self.entity_name}] Loading Regex patterns...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Αγνοούμε κενές γραμμές και σχόλια
                if not line or line.startswith('#'):
                    continue
                try:
                    # Προσθήκη του pattern στη λίστα
                    self.patterns.append(re.compile(line, re.IGNORECASE))
                except re.error as e:
                    print(f"Warning: Invalid regex '{line}': {e}")

    def _load_lexicon(self, path):
        print(f"[{self.entity_name}] Loading Lexicon...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Λίστα με τα keys που περιέχουν ΒΟΗΘΗΤΙΚΕΣ λέξεις και ΠΡΕΠΕΙ να αγνοηθούν.
            # (Αυτά τα keys τα εντοπίσαμε στο enhanced_date_lexicon.json)
            EXCLUDED_KEYS = ['date_prepositions', 'temporal_relations', 
                             'legal_date_contexts', 'time_prepositions']
            
            # Περίπτωση 1: Το JSON είναι απλή λίστα (π.χ. facility_lexicon.json)
            if isinstance(data, list):
                # Φορτώνουμε όλη τη λίστα, καθώς υποθέτουμε ότι περιέχει entities
                self.lexicon_phrases.update([p.lower() for p in data])
            
            # Περίπτωση 2: Το JSON είναι λεξικό με κατηγορίες (π.χ. enhanced_date_lexicon.json)
            elif isinstance(data, dict):
                # Φορτώνουμε μόνο τα keys που αφορούν ΟΝΤΟΤΗΤΕΣ (μηνύες, ημέρες, κτλ)
                for category, words in data.items():
                    if category not in EXCLUDED_KEYS and isinstance(words, list):
                        # Προσθήκη ολόκληρης της φράσης (π.χ. "Ιανουαρίου")
                        self.lexicon_phrases.update([str(w).lower() for w in words])

    def predict(self, text):
        results = []
        
        # 1. Έλεγχος με Regex
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                results.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": self.entity_name,
                    "text": match.group(),
                    "source": "Regex",
                    "confidence": 1.0
                })
        
        # 2. Έλεγχος με Lexicon (Exact Match - Απλοϊκή υλοποίηση για τώρα)
        # Σημείωση: Για μεγάλα κείμενα θα βάλουμε FlashText αργότερα
        lower_text = text.lower()
        for phrase in self.lexicon_phrases:
            if phrase in lower_text:
                # Βρες τη θέση (προσοχή: βρίσκει μόνο την πρώτη εμφάνιση έτσι - θέλει βελτίωση για production)
                start = lower_text.find(phrase)
                while start != -1:
                    results.append({
                        "start": start,
                        "end": start + len(phrase),
                        "label": self.entity_name,
                        "text": text[start:start+len(phrase)],
                        "source": "Lexicon",
                        "confidence": 1.0
                    })
                    start = lower_text.find(phrase, start + 1)
                    
        return results