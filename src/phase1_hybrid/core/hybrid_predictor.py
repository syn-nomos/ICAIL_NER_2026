
import numpy as np
import torch
from typing import List, Dict, Any, Optional

class HybridPredictor:
    def __init__(self, roberta_model, vector_memory, attention_extractor, embedding_builder, rule_agents=None, llm_judge=None):
        """
        The Brain of the TAAL System.
        Combines Neural (RoBERTa), Symbolic (Triggers), Memory (Vectors) AND Rules (Regex/Lexicons).
        """
        self.roberta = roberta_model
        self.memory = vector_memory
        self.attention = attention_extractor
        self.emb_builder = embedding_builder
        self.rule_agents = rule_agents if rule_agents else []
        self.llm_judge = llm_judge # Instance that has .resolve(text, entities) -> entities
        
        # Weights for the Hybrid Formula
        self.W_MEM_VECTOR = 0.5
        self.W_MEM_TRIGGER = 0.3
        self.W_MODEL_CONF = 0.2

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Main Pipeline (The "Decision Stack"):
        1. Level 1: Rule Engine (Regex/Lexicons) - High Confidence
        2. Level 2 & 3: Memory + RoBERTa (Hybrid)
        3. Level 4: Conflict Resolution (merging)
        4. Level 5: LLM (if needed - e.g. conflict)
        """
        all_candidates = []

        # --- LEVEL 1: RULES ---
        for agent in self.rule_agents:
            try:
                rule_hits = agent.predict(text)
                for hit in rule_hits:
                    hit['source'] = f"Rule:{agent.__class__.__name__}"
                    hit['explanation'] = "Pattern Match"
                    hit['trigger_text'] = None # Rules don't use triggers usually
                    all_candidates.append(hit)
            except Exception as e:
                print(f"Rule Agent Error: {e}")

        # --- LEVEL 2 & 3: MODEL + MEMORY ---
        # 1. Base Prediction (RoBERTa)
        try:
            raw_entities = self.roberta.predict(text)
        except Exception as e:
            print(f"RoBERTa Predict Error: {e}")
            raw_entities = []
        
        enriched_neural_entities = []
        for ent in raw_entities:
            # Copy basic info
            result = ent.copy()
            result['source'] = 'RoBERTa'
            result['explanation'] = "Neural Prediction"

            # 2. Extract Trigger
            triggers = self.attention.extract_triggers(
                self.roberta,
                text=text, 
                entity_start_char=ent['start'], 
                entity_end_char=ent['end'],
                top_k=1
            )
            best_trigger = triggers[0] if triggers else None
            
            if best_trigger:
                result['trigger_text'] = best_trigger.text
                result['trigger_score'] = best_trigger.score
                result['trigger_direction'] = best_trigger.direction
            else:
                result['trigger_text'] = None

            # 3. Memory Lookup
            aug_vector = self.emb_builder.build(self.roberta, text, ent['start'], ent['end'], ent['label'])
            result['vector_blob'] = aug_vector.tobytes()

            matches = self.memory.find_similar(aug_vector, k=3, threshold=0.7)
            
            if matches:
                top_match = matches[0]
                mem_score = top_match['similarity']
                # Determine trigger boost
                trigger_boost = 0.0
                if best_trigger and top_match.get('trigger_text'):
                    if top_match['trigger_text'] == best_trigger.text:
                        trigger_boost = 0.15
                
                final_confidence = (0.7 * mem_score) + (0.3 * ent['confidence']) + trigger_boost
                final_confidence = min(final_confidence, 1.0)
                
                result['confidence'] = final_confidence
                
                # Robust key access for 'text_span' vs 'text'
                matched_text = top_match.get('text_span', top_match.get('text', 'Unknown'))
                
                result['memory_match'] = {
                    'text': matched_text,
                    'similarity': mem_score,
                    'trigger': top_match.get('trigger_text')
                }
                result['source'] = 'Hybrid (Memory)'
                result['explanation'] = f"Recall '{matched_text}'"
            else:
                result['memory_match'] = None

            enriched_neural_entities.append(result)

        # Merge Neural into Candidates
        all_candidates.extend(enriched_neural_entities)

        # --- LEVEL 4: MERGE & RESOLVE ---
        # Simple strategy: Prefer Rules over Neural if they overlap significantly
        final_entities = self._merge_overlapping(all_candidates)

        return final_entities

    def _merge_overlapping(self, candidates: List[Dict]) -> List[Dict]:
        """
        Greedy Non-Maximum Suppression (NMS) with "Longest Match Prevails" logic.
        Refined for User Requirement: "Larger boundaries should prevail" if they envelop smaller ones.
        """
        # Primary Sort: Confidence. Secondary: Length (Longer first).
        # This helps break ties, but doesn't solve 1.0 vs 0.95 case alone.
        candidates.sort(key=lambda x: (x.get('confidence', 0.0), x['end'] - x['start']), reverse=True)
        
        kept = []
        for cand in candidates:
            c_start, c_end = cand['start'], cand['end']
            
            # Action flags
            is_suppressed = False
            idx_to_replace = -1
            
            for i, k in enumerate(kept):
                k_start, k_end = k['start'], k['end']
                
                # Check Intersection
                if max(c_start, k_start) < min(c_end, k_end):
                    
                    # Logic 1: If we found an overlap with an existing 'kept' entity.
                    # Usually, 'k' has higher confidence (due to sort order).
                    
                    # Logic 2: CONTAINMENT OVERRIDE
                    # If 'cand' (New) fully contains 'k' (Existing), and 'cand' is reasonably confident,
                    # we prefer 'cand' even if 'k' had slightly higher confidence.
                    # Example: k="v. 4000" (Conf 1.0), cand="v. 4000/2012" (Conf 0.9).
                    
                    len_c = c_end - c_start
                    len_k = k_end - k_start
                    
                    cand_envelops_k = (c_start <= k_start) and (c_end >= k_end) and (len_c > len_k)
                    
                    if cand_envelops_k:
                        # Safety: Only override if the confidence drop isn't catastrophic
                        # e.g. Don't replace 1.0 with 0.1 just because it's longer.
                        if cand['confidence'] >= 0.7: 
                            idx_to_replace = i
                            # We found a victim. Stop looking for conflicts (assumption: cand dominant)
                            break
                    
                    # Default: Suppress 'cand' because 'k' is better/earlier
                    is_suppressed = True
                    break
            
            if idx_to_replace != -1:
                # Replace the smaller/shorter entity with the larger one
                kept.pop(idx_to_replace)
                kept.append(cand)
                # Re-sort might be needed strictly speaking, but for NMS valid-set it's okay.
                
            elif not is_suppressed:
                kept.append(cand)
        
        # Final Sort by Position
        kept.sort(key=lambda x: x['start'])
        return kept
        kept.sort(key=lambda x: x['start'])
        return kept
