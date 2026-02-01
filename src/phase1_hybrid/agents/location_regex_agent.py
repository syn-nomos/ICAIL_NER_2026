import re
from pathlib import Path
from typing import List, Dict, Any

class LocationRegexAgent:
    def __init__(self, regex_path=None):
        if regex_path is None:
            regex_path = Path("data/knowledge_base/LOCATION/patterns.txt")
        
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
                                patterns.append(re.compile(line))
                            except re.error as e:
                                print(f"⚠️ Invalid Location Regex: {line} -> {e}")
            except Exception as e:
                print(f"❌ Error loading LOCATION patterns: {e}")
        return patterns

    def predict(self, text: str) -> List[Dict[str, Any]]:
        results = []
        used_positions = set()
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                # SUPPORT FOR CAPTURING GROUPS:
                # If regex has groups, find the first non-None group.
                if match.groups():
                    target_group_index = next((i for i, g in enumerate(match.groups()) if g is not None), None)
                    if target_group_index is not None:
                        # Group indices are 1-based in group()
                        match_text = match.group(target_group_index + 1)
                        start = match.start(target_group_index + 1)
                        end = match.end(target_group_index + 1)
                    else:
                         # Should ideally not happen if match occurred, but fallback
                        match_text = match.group()
                        start = match.start()
                        end = match.end()
                else:
                    match_text = match.group()
                    start = match.start()
                    end = match.end()
                
                # Filter: Ignore very short matches or empty groups
                if not match_text or len(match_text) < 3: continue

                if any(pos in used_positions for pos in range(start, end)):
                    continue
                
                for pos in range(start, end):
                    used_positions.add(pos)
                
                results.append({
                    "text": match_text,
                    "label": "LOCATION",
                    "start": start,
                    "end": end,
                    "confidence": 0.92, # Αρκετά υψηλό, ειδικά για 'Νήσος'/'Όρος'
                    "source": "Regex"
                })
        
        return results