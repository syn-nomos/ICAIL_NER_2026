from typing import List, Dict, Any, Set, Tuple
import os
from src.judges.llm_client import LLMJudge, CloudScanner
from src.core.hybrid_predictor import HybridPredictor
import difflib

class Council:
    """
    The AI Council: A 3-Tier Architecture for High-Fidelity Annotation.
    
    Tier 0: Local "Worker" (HybridPredictor: RoBERTa + Rules) - Note: Meltemi integration is optional.
    Tier 1: Cloud "Scanner" (DeepSeek V3 / GPT-4o-mini - High Recall)
    Tier 2: Cloud "Judge" (DeepSeek R1 / Claude 3.5 - Logic & Reasoning)
    """
    def __init__(self, hybrid_predictor: HybridPredictor, llm_judge: LLMJudge):
        self.tier0 = hybrid_predictor
        self.tier1 = CloudScanner()              # Updated to CloudScanner (OpenRouter)
        self.tier2 = llm_judge                   # Existing Judge (DeepSeek R1)
        
    def convene(self, text: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Runs the full council session on a text.
        Returns the FINAL agreed list of entities.
        """
        
        # --- 1. Tier 0 Execution (Local: RoBERTa + Rules) ---
        tier0_preds = self.tier0.predict(text)
        
        # --- 🚀 FAST PASS (Turbo Mode) ---
        # If we have strong local results (Rules or High Confidence RoBERTa), 
        # we TRUST them and skip the slow Cloud Scanner/Judge.
        # This reduces processing time from ~20s to ~0.5s per sentence.
        
        high_confidence_only = True
        if not tier0_preds:
            high_confidence_only = False
        else:
            for p in tier0_preds:
                # Rule-based or RoBERTa > 0.95 (Very Confident)
                is_rule = 'Rule' in p.get('source', '')
                is_confident = p.get('score', 0) > 0.95
                
                if not (is_rule or is_confident):
                    high_confidence_only = False
                    break
        
        if high_confidence_only:
            # ✅ TRUST LOCAL: Auto-accept everything
            for p in tier0_preds:
                tag = "Rule" if 'Rule' in p.get('source', '') else "Local"
                p['council_status'] = f"FAST_PASS ({tag})"
                p['is_accepted'] = True
            return self._resolve_overlaps(tier0_preds)
            
        # [ULTRA FAST MODE OVERRIDE] - DISABLED for Validation Run
        # User requested max speed. Even if confidence is low, strictly trust Local Tier 0.
        # Skip Tier 1 (Cloud Scanner) entirely.
        
        # Check conflicts purely internal to Tier 0 (e.g. Regex vs RoBERTa same span)
        # But usually HybridPredictor handles that internally.
        
        # for p in tier0_preds:
        #      p['council_status'] = "AUTO_ACCEPTED (Bulk Mode)"
        #      p['is_accepted'] = True
             
        # return self._resolve_overlaps(tier0_preds)

        # --- 2. Tier 1 Execution (Cloud Scanner) ---
        # SKIPPED FOR SPEED -> ACTIVATED for Validation
        try:
             tier1_preds = self.tier1.scan_text(text)
        except Exception as e:
             print(f"Tier 1 Error: {e}")
             tier1_preds = []
        
        # [CRITICAL] Inject Start/End for Tier 1 (Scanner) to enable Overlap Check
        for e in tier1_preds:
            # DEFENSIVE NORMALIZATION
            if e['label'] == 'LEG_REFS': e['label'] = 'LEG-REFS'
            if e['label'] == 'public-docs': e['label'] = 'PUBLIC-DOCS'
            
            if 'start' not in e:
                s, end = self._locate_idx(text, e['text'])
                e['start'] = s
                e['end'] = end

        # --- 3. Unification & Discovery ---
        final_entities = []
        
        # Convert lists to manageable dicts mapping text -> entity
        # We normalize text for comparison
        t0_map = {self._normalize(e['text']): e for e in tier0_preds}
        t1_map = {self._normalize(e['text']): e for e in tier1_preds}
        
        all_keys = set(t0_map.keys()) | set(t1_map.keys())
        
        for key in all_keys:
            ent0 = t0_map.get(key)
            ent1 = t1_map.get(key)
            
            # Case A: Platinum Agreement (Both Agreement)
            if ent0 and ent1:
                if ent0['label'] == ent1['label']:
                    # UNANIMOUS AGREEMENT -> AUTO ACCEPT
                    ent0['council_status'] = 'PLATINUM (Unanimous)'
                    ent0['is_accepted'] = True
                    final_entities.append(ent0)
                else:
                    # CONFLICT (Label mismatch) -> TIER 2 JUDGE
                    decision = self.tier2.resolve_conflict(text, [ent0, ent1])
                    winner_idx = decision.get('best_option_index', 1)
                    winner = ent0 if winner_idx == 1 else ent1
                    winner['council_status'] = f"JUDGED (Conflict: {ent0['label']} vs {ent1['label']})"
                    winner['is_accepted'] = True
                    final_entities.append(winner)
                    
            # Case B: Tier 0 Only
            elif ent0:
                # If it comes from a RULE, we trust it absolutely.
                if 'Rule:' in ent0.get('source', ''):
                    ent0['council_status'] = 'PLATINUM (Rule)'
                    ent0['is_accepted'] = True
                    final_entities.append(ent0)
                else:
                    # RoBERTa Only.
                    if ent0.get('score', 0) > 0.95:
                        ent0['council_status'] = 'HIGH_CONFIDENCE (Local)'
                        ent0['is_accepted'] = True
                        final_entities.append(ent0)
                    else:
                        # JUDGE IT.
                        validation = self.tier2.validate_entity(text, ent0['text'], ent0['label'])
                        if validation.get('is_valid'):
                            ent0['council_status'] = 'JUDGED (Valid Local)'
                            ent0['is_accepted'] = True
                            final_entities.append(ent0)
                        else:
                            print(f"     ❌ REJECTED: '{ent0['text']}' ({ent0['label']})")
            
            # Case C: Tier 1 Only (Discovery - High Recall)
            elif ent1:
                # Must be verified by Judge.
                validation = self.tier2.validate_entity(text, ent1['text'], ent1['label'])
                if validation.get('is_valid'):
                    ent1['council_status'] = 'DISCOVERY (Scanner + Judge)'
                    ent1['score'] = 1.0 
                    ent1['source'] = 'Council'
                    ent1['is_accepted'] = True
                    # Only add if we found it in text (start != -1)
                    if ent1.get('start', -1) != -1:
                        final_entities.append(ent1)
                else:
                    print(f"     ❌ REJECTED: '{ent1['text']}' ({ent1['label']})")
        
        # --- 4. Overlap Resolution ---
        resolved_entities = self._resolve_overlaps(final_entities)
        
        # --- 5. BOUNDARY REFINEMENT (The "3 Agents" Logic) ---
        # Automatically fix common errors for specific types
        for ent in resolved_entities:
            # We refine everything except Rule-based ones (which are usually strict Regex)
            # OR we can refine everything. Let's refine scanned/judged ones mainly.
            if 'Rule' not in ent.get('council_status', ''):
                lbl = ent['label']
                # Call Refine Limit from Tier 2 (LLM Client has the logic)
                refined_text = self.tier2.refine_boundaries(text, ent['text'], lbl)
                
                if refined_text != ent['text']:
                    print(f"     ✂️ Refined: '{ent['text']}' -> '{refined_text}'")
                    ent['text'] = refined_text
                    # Re-calc start/end
                    s, e = self._locate_idx(text, refined_text)
                    if s != -1:
                        ent['start'] = s
                        ent['end'] = e

        return resolved_entities

    def _locate_idx(self, text: str, chunk: str) -> Tuple[int, int]:
        """Finds the first occurrence of chunk in text (case-insensitive fallback)."""
        if not chunk: return -1, -1
        # 1. Exact match
        idx = text.find(chunk)
        if idx != -1: return idx, idx + len(chunk)
        
        # 2. Case Insensitive match
        import re
        try:
            m = re.search(re.escape(chunk), text, re.IGNORECASE)
            if m: return m.start(), m.end()
        except:
            pass
        return -1, -1

    def _resolve_overlaps(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes overlapping entities using a Priority System.
        Priority: PLATINUM > JUDGED > HIGH_CONFIDENCE > DISCOVERY.
        Tie-Breaker: Length (Longest Match Wins).
        """
        if not entities: return []

        def get_priority(e):
            status = e.get('council_status', '')
            if 'Unanimous' in status: return 1
            if 'Rule' in status: return 2
            if 'JUDGED' in status: return 3
            if 'Valid Local' in status: return 4
            if 'DISCOVERY' in status: return 5
            return 99

        # Sort: Better Priority (Lower) -> Longer Length (Higher) -> Higher Score
        entities.sort(key=lambda x: (
            get_priority(x), 
            -len(x['text']), 
            -x.get('score', 0)
        ))

        kept = []
        for cand in entities:
            c_start, c_end = cand.get('start', -1), cand.get('end', -1)
            if c_start == -1: continue # Skip locatable errors
            
            is_overlap = False
            for k in kept:
                k_start, k_end = k.get('start', -1), k.get('end', -1)
                # Overlap Logic: Ends after Start AND Starts before End
                if (c_start < k_end) and (c_end > k_start):
                    is_overlap = True
                    # Debug Log if needed
                    # print(f"Overlap Removed: '{cand['text']}' ({cand['council_status']}) by '{k['text']}' ({k['council_status']})")
                    break
            
            if not is_overlap:
                kept.append(cand)
                
        return kept

    def _normalize(self, s: str) -> str:
        """Simple normalization for matching"""
        return s.strip().lower()
