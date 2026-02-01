import re
import math
from typing import List, Dict, Any
from src.judges.llm_client import LLMJudge

class NerController:
    """
    V4 Controller: Strict Filtering & LLM Routing.
    Δεν αφήνει σκουπίδια. Αν έχει αμφιβολία, μαρκάρει για LLM Review.
    """
    def __init__(self, memory=None, model_wrapper=None):
        self.llm_judge = LLMJudge()
        self.memory = memory
        self.model_wrapper = model_wrapper
        
        # Βάρη Πηγών (Reliability)
        self.source_reliability = {
            "Lexicon": 0.99,       
            "Lexicon_VIP": 0.99,   
            "Regex": 0.95,         
            "Hybrid": 0.85,        # Μειώσαμε λίγο την εμπιστοσύνη στο Hybrid λόγω των θραυσμάτων
            "RoBERTa": 0.75,       
            "Agent": 0.70          
        }
        
        # Blacklist (Stopwords & Garbage)
        self.blacklist = {
            "και", "του", "την", "τον", "των", "τους", "τις", "μια", "ένα", "στις", "στους",
            "από", "προς", "με", "σε", "για", "ως", "κατά", "δια", "υπό", "επί",
            "απόφαση", "νόμος", "άρθρο", "παράγραφος", "τεύχος", "φεκ", "αριθμ", "αριθ",
            "υπουργείο", "διεύθυνση", "τμήμα", "γραφείο", "σώμα", "επιτροπή", 
            "ελλάδα", "αθήνα", "θεσσαλονίκη", "η", "ο", "το", "τα", "οι", "αι", "δεν", "θα", "να",
            "έα", "οί", "ορ", "ν.", "π.δ." # Προσθέσαμε κοινά θραύσματα
        }

    def resolve(self, candidates: List[Dict[str, Any]], text_context: str = "") -> List[Dict[str, Any]]:
        valid_candidates = []
        
        for cand in candidates:
            # Normalize keys
            if 'start_char' in cand: cand['start'] = cand.pop('start_char')
            if 'end_char' in cand: cand['end'] = cand.pop('end_char')
            
            # --- SNAP TO WORD BOUNDARIES (User Request) ---
            # Διασφαλίζει ότι δεν κόβουμε λέξεις στη μέση (π.χ. "Δωδ" -> "Δωδεκανήσου")
            self._snap_to_word_boundaries(cand, text_context)

            # --- STRICT VALIDATION ---
            if self._is_garbage(cand):
                # print(f"🗑️ Garbage Rejected: {cand['text']}")
                continue

            # --- VECTOR MEMORY CHECK (Active Learning Integration) ---
            if self.memory and self.model_wrapper:
                self._apply_vector_memory_logic(cand, text_context)
            
            # Υπολογισμός Σκορ
            cand['final_score'] = self._calculate_score(cand)
            
            # --- LOW SCORE FILTERING ---
            # Αν το τελικό σκορ είναι χάλια, το πετάμε πριν καν μπει στη βάση.
            if cand['final_score'] < 0.45:
                # print(f"📉 Low Score Rejected: {cand['text']} ({cand['final_score']})")
                continue
            
            # --- LLM VALIDATION FOR GRAY ZONE ---
            # Αν είναι μεταξύ 0.45 και 0.60, ρωτάμε τον LLM αν είναι έγκυρο
            if 0.45 <= cand['final_score'] < 0.60:
                print(f"⚖️  Validating Gray Zone Entity: '{cand['text']}' ({cand['final_score']})")
                validation_result = self.llm_judge.validate_entity(text_context, cand['text'], cand['label'])
                
                if validation_result['is_valid']:
                    print(f"✅ LLM Approved: '{cand['text']}'")
                elif validation_result['boundary_error']:
                    print(f"✂️  LLM Detected Boundary Error: '{cand['text']}'")
                    # Call refiner
                    refined_text = self.llm_judge.refine_boundaries(text_context, cand['text'], cand['label'])
                    
                    if refined_text != cand['text'] and refined_text in text_context:
                         # Find new start/end
                         search_start = max(0, cand['start'] - 10)
                         search_end = min(len(text_context), cand['end'] + 10)
                         local_context = text_context[search_start:search_end]
                         
                         idx = local_context.find(refined_text)
                         if idx != -1:
                             new_start = search_start + idx
                             new_end = new_start + len(refined_text)
                             print(f"✨ Refined to: '{refined_text}'")
                             cand['start'] = new_start
                             cand['end'] = new_end
                             cand['text'] = refined_text
                             cand['resolution_type'] = 'LLM_GrayZone_Refinement'
                             # Boost score slightly to keep it
                             cand['final_score'] = 0.65 
                         else:
                             print(f"❌ LLM Could not locate refined text: '{refined_text}'")
                             continue
                    else:
                         print(f"❌ LLM Could not refine: '{cand['text']}'")
                         continue
                else:
                    print(f"❌ LLM Rejected: '{cand['text']}'")
                    continue

            valid_candidates.append(cand)

        # --- BOUNDARY CORRECTION (NEW) ---
        # 1. Snap-to-Precision: RoBERTa snaps to Regex/Lexicon
        # 2. LLM Refiner: RoBERTa gets trimmed by LLM if alone
        valid_candidates = self._correct_boundaries(valid_candidates, text_context)
        # ---------------------------------

        # Conflict Resolution (Greedy with LLM Fallback)
        valid_candidates.sort(key=lambda x: (x['final_score'], len(x['text'])), reverse=True)

        final_entities = []
        occupied_mask = [False] * len(text_context)

        # Group overlapping candidates
        clusters = self._cluster_candidates(valid_candidates)

        for cluster in clusters:
            if len(cluster) == 1:
                # No conflict
                cand = cluster[0]

                # --- NEW: Single High-Risk Audit (GPE vs LOCATION) ---
                # Αν έχουμε μεμονωμένη οντότητα Roberta (Score < 0.85) που είναι GPE/LOCATION,
                # κάνουμε Semantic Check γιατί το μοντέλο μπερδεύεται συχνά.
                if cand['label'] in ['GPE', 'LOCATION'] and 0.60 <= cand['final_score'] < 0.85:
                    print(f"🔍 Auditing Single High-Risk Entity: '{cand['text']}' ({cand['label']})")
                    corrected_label = self.llm_judge.evaluate_ambiguity(text_context, cand['text'], "GPE", "LOCATION")
                    
                    if corrected_label and corrected_label != cand['label']:
                        print(f"🔄 Correction: '{cand['text']}' {cand['label']} -> {corrected_label}")
                        cand['label'] = corrected_label
                        cand['resolution_type'] = "LLM_HighRisk_Correction"
                        cand['final_score'] = max(cand['final_score'], 0.85) # Boost confidence
                # -----------------------------------------------------

                if self._is_free(occupied_mask, cand['start'], cand['end']):
                    self._occupy(occupied_mask, cand['start'], cand['end'])
                    final_entities.append(cand)
            else:
                # Conflict!
                # Check for specific ambiguities (GPE vs LOCATION, ORG vs FACILITY)
                labels = {c['label'] for c in cluster}
                
                if "GPE" in labels and "LOCATION" in labels:
                    # Ambiguity: GPE vs LOCATION
                    # Pick the best candidate text (longest usually)
                    best_cand = max(cluster, key=lambda x: len(x['text']))
                    print(f"🤔 Ambiguity Detected (GPE vs LOCATION): '{best_cand['text']}'")
                    resolved_label = self.llm_judge.evaluate_ambiguity(text_context, best_cand['text'], "GPE", "LOCATION")
                    
                    if resolved_label:
                        print(f"💡 LLM Resolved to: {resolved_label}")
                        best_cand['label'] = resolved_label
                        if self._is_free(occupied_mask, best_cand['start'], best_cand['end']):
                            # --- AGGREGATE SOURCES ---
                            contributing_sources = set()
                            for c in cluster:
                                if c['text'] == best_cand['text']: # Label might have changed, so just check text
                                    contributing_sources.add(c['source'])
                            best_cand['source'] = ", ".join(sorted(list(contributing_sources)))
                            best_cand['resolution_type'] = "LLM_Ambiguity_Resolution"
                            # -------------------------
                            
                            self._occupy(occupied_mask, best_cand['start'], best_cand['end'])
                            final_entities.append(best_cand)
                        continue

                # General Conflict Resolution via LLM
                # Only if scores are close (e.g. top 2 diff < 0.1)
                top_2 = sorted(cluster, key=lambda x: x['final_score'], reverse=True)[:2]
                if len(top_2) == 2 and (top_2[0]['final_score'] - top_2[1]['final_score'] < 0.15):
                    print(f"⚔️  Conflict Detected: '{top_2[0]['text']}' vs '{top_2[1]['text']}'")
                    # Call LLM to decide between top candidates
                    try:
                        decision = self.llm_judge.resolve_conflict(text_context, top_2)
                        
                        # Handle case where LLM returns None or invalid JSON structure
                        best_idx = 0
                        if decision and isinstance(decision, dict):
                            best_idx = decision.get('best_option_index')
                            # Ensure it's an integer
                            if not isinstance(best_idx, int):
                                best_idx = 0
                        
                        if best_idx > 0:
                            winner_idx = best_idx - 1
                            if winner_idx < len(top_2):
                                winner = top_2[winner_idx]
                                print(f"🏆 LLM Winner: '{winner['text']}'")
                                
                                # --- AGGREGATE SOURCES ---
                                contributing_sources = set()
                                for c in cluster:
                                    if c['text'] == winner['text'] and c['label'] == winner['label']:
                                        contributing_sources.add(c['source'])
                                winner['source'] = ", ".join(sorted(list(contributing_sources)))
                                winner['resolution_type'] = "LLM_Conflict_Resolution"
                                # -------------------------
    
                                if self._is_free(occupied_mask, winner['start'], winner['end']):
                                    self._occupy(occupied_mask, winner['start'], winner['end'])
                                    final_entities.append(winner)
                                continue
                    except Exception as e:
                       print(f"⚠️ LLM Conflict Resolution Failed: {e}")
                       # Fallback to score
                       pass
                
                # Fallback to Greedy (Highest Score)
                winner = cluster[0] # Already sorted by score
                
                # --- AGGREGATE SOURCES ---
                # Collect all sources that proposed this EXACT winner (same text and label)
                contributing_sources = set()
                for c in cluster:
                    if c['text'] == winner['text'] and c['label'] == winner['label']:
                        contributing_sources.add(c['source'])
                
                # Update the winner's source field
                winner['source'] = ", ".join(sorted(list(contributing_sources)))
                winner['resolution_type'] = "Score_Based" if len(cluster) > 1 else "Direct"
                # -------------------------

                if self._is_free(occupied_mask, winner['start'], winner['end']):
                    self._occupy(occupied_mask, winner['start'], winner['end'])
                    final_entities.append(winner)

        final_entities.sort(key=lambda x: x['start'])
        return final_entities

    def _cluster_candidates(self, candidates):
        """Groups overlapping candidates into clusters."""
        if not candidates: return []
        # Sort by start time
        candidates.sort(key=lambda x: x['start'])
        
        clusters = []
        current_cluster = [candidates[0]]
        cluster_end = candidates[0]['end']
        
        for i in range(1, len(candidates)):
            cand = candidates[i]
            if cand['start'] < cluster_end:
                current_cluster.append(cand)
                cluster_end = max(cluster_end, cand['end'])
            else:
                clusters.append(current_cluster)
                current_cluster = [cand]
                cluster_end = cand['end']
        clusters.append(current_cluster)
        return clusters

    def _is_free(self, mask, start, end):
        if start < 0 or end > len(mask): return False
        for i in range(start, end):
            if mask[i]: return False
        return True

    def _occupy(self, mask, start, end):
        for i in range(start, end):
            mask[i] = True

    def _correct_boundaries(self, candidates: List[Dict[str, Any]], text_context: str) -> List[Dict[str, Any]]:
        """
        Διορθώνει τα όρια των οντοτήτων.
        1. Snap-to-Precision: Το RoBERTa 'κουμπώνει' στα όρια των Regex/Lexicon.
        2. LLM Refiner: Το RoBERTa καθαρίζεται από τον LLM αν είναι μόνο του.
        """
        trusted_sources = {'Regex', 'Lexicon', 'Lexicon_VIP'}
        
        # Separate candidates
        trusted_cands = [c for c in candidates if c['source'] in trusted_sources]
        model_cands = [c for c in candidates if c['source'] not in trusted_sources]
        
        final_cands = trusted_cands[:] # Keep trusted as is
        
        for mc in model_cands:
            snapped = False
            # 1. Try to snap to a trusted candidate
            for tc in trusted_cands:
                # Check overlap
                if max(mc['start'], tc['start']) < min(mc['end'], tc['end']):
                    # Overlap found! Snap to trusted boundaries
                    # print(f"🧲 Snapping '{mc['text']}' -> '{tc['text']}'")
                    mc['start'] = tc['start']
                    mc['end'] = tc['end']
                    mc['text'] = tc['text']
                    mc['source'] += f", {tc['source']}" # Merge sources
                    snapped = True
                    break # Snap to the first one found (usually only one)
            
            if snapped:
                # Add the snapped candidate (avoid duplicates if possible, but controller handles overlaps later)
                final_cands.append(mc)
            else:
                # 2. LLM Refiner (Only for RoBERTa if it looks suspicious)
                # Suspicious: Starts/Ends with non-alphanumeric or common stopwords
                # User Request: Don't refine very long entities (likely titles) to avoid cutting them.
                text = mc['text']
                is_suspicious = (not text[0].isalnum() or not text[-1].isalnum() or 
                                 text.split()[0].lower() in self.blacklist or 
                                 text.split()[-1].lower() in self.blacklist)
                
                if is_suspicious and len(text) < 100: # Limit refinement to shorter entities
                    print(f"🧹 Refining Boundaries for: '{text}'")
                    refined_text = self.llm_judge.refine_boundaries(text_context, text, mc['label'])
                    
                    if refined_text != text and refined_text in text_context:
                        # Find new start/end
                        # We search near the original position to avoid finding the same word elsewhere
                        search_start = max(0, mc['start'] - 5)
                        search_end = min(len(text_context), mc['end'] + 5)
                        local_context = text_context[search_start:search_end]
                        
                        idx = local_context.find(refined_text)
                        if idx != -1:
                            new_start = search_start + idx
                            new_end = new_start + len(refined_text)
                            print(f"✨ Refined: '{text}' -> '{refined_text}'")
                            mc['start'] = new_start
                            mc['end'] = new_end
                            mc['text'] = refined_text
                            mc['resolution_type'] = "LLM_Boundary_Refinement"
                
                final_cands.append(mc)
                
        return final_cands

    def _apply_vector_memory_logic(self, cand: Dict[str, Any], text: str):
        """
        Ελέγχει αν η υποψήφια οντότητα υπάρχει στη μνήμη χρησιμοποιώντας Weighted k-NN (k=3).
        Αυξάνει την αξιοπιστία της μνήμης μέσω ψηφοφορίας.
        """
        try:
            # 1. Generate Vector on-the-fly (if missing)
            if 'vector' not in cand or cand['vector'] is None:
                # Create a mini-batch with just this candidate
                temp_span = [{'text': cand['text'], 'start': cand['start'], 'end': cand['end'], 'label': cand['label']}]
                enriched = self.model_wrapper.enrich_spans_with_vectors(text, temp_span)
                if enriched and enriched[0].get('vector') is not None:
                    cand['vector'] = enriched[0]['vector']
            
            if cand.get('vector') is None: return

            # 2. Query Positive Memory (Weighted k-NN, k=3)
            # Χαμηλώνουμε λίγο το threshold (0.88) για να πιάσουμε 3 γείτονες, αλλά κρίνουμε αυστηρά.
            matches = self.memory.find_similar(cand['vector'], k=3, threshold=0.88)
            
            if matches:
                # Υπολογισμός Ψήφων (Weighted by Similarity)
                votes = {}
                for m in matches:
                    lbl = m['label']
                    sim = m['similarity']
                    votes[lbl] = votes.get(lbl, 0) + sim # Add similarity score as weight
                
                # Βρες τον νικητή
                best_label = max(votes, key=votes.get)
                total_weight = votes[best_label]
                supporting_neighbors = sum(1 for m in matches if m['label'] == best_label)
                avg_sim = total_weight / supporting_neighbors
                
                # --- LOGIC DECISION MATRIX ---
                
                # CASE A: Strong Consensus (π.χ. 2 ή 3 γείτονες συμφωνούν)
                if supporting_neighbors >= 2:
                    if best_label == cand['label']:
                        # Επιβεβαίωση (Validation)
                        print(f"🧠 Memory Consensus ({supporting_neighbors}/3): Confirmed '{cand['text']}' as {best_label}")
                        cand['confidence'] = 1.0
                        cand['source'] += f"_MemVoc({supporting_neighbors})"
                    else:
                        # Διόρθωση (Correction) - Μόνο αν η μέση ομοιότητα είναι πολύ υψηλή
                        if avg_sim > 0.92:
                            print(f"🧠 Memory Correction ({supporting_neighbors}/3): '{cand['text']}' {cand['label']} -> {best_label}")
                            cand['label'] = best_label
                            cand['confidence'] = 0.98
                            cand['source'] += f"_MemCorr({supporting_neighbors})"
                
                # CASE B: Single Strong Match (1-NN Fallback for rare entities)
                elif supporting_neighbors == 1 and avg_sim > 0.96:
                    if best_label == cand['label']:
                        cand['confidence'] = 1.0
                        cand['source'] += "_MemExact"
                    elif avg_sim > 0.97: # Πολύ αυστηρό για διόρθωση βάσει ενός μόνο δείγματος
                         cand['label'] = best_label
                         # cand['confidence'] = 0.95 # Αφήνουμε το score ως έχει ή το ανεβάζουμε λίγο

            # 3. Query Negative Memory (Rejected) - Εδώ αρκεί το 1-NN (Iron Dome)
            # Αν κάτι μοιάζει ΤΟΣΟ πολύ (0.96) με κάτι που απορρίψαμε, το κόβουμε.
            if hasattr(self.memory, 'rejected_vectors') and self.memory.rejected_vectors is not None:
                is_rejected, rejected_text_match = self.memory.check_is_rejected(cand['vector'], cand['label'], threshold=0.96)
                if is_rejected:
                    print(f"🛡️ Negative Memory hit: '{cand['text']}' matches rejected '{rejected_text_match}'")
                    cand['confidence'] = 0.0 # Kill candidate
                    cand['source'] += "_MemRejected"
                    return 

        except Exception as e:
            # print(f"⚠️ Memory Lookup Error: {e}")
            pass

    def _snap_to_word_boundaries(self, cand: Dict[str, Any], text: str):
        """
        Επεκτείνει τα όρια της οντότητας ώστε να συμπεριλάβει ολόκληρες λέξεις.
        Π.χ. "Νομού Δωδ" -> "Νομού Δωδεκανήσου"
        """
        start = cand['start']
        end = cand['end']
        
        # Expand Left
        while start > 0 and text[start-1].isalnum():
            start -= 1
            
        # Expand Right
        while end < len(text) and text[end].isalnum():
            end += 1
            
        if start != cand['start'] or end != cand['end']:
            new_text = text[start:end]
            # print(f"🔧 Snapped '{cand['text']}' -> '{new_text}'")
            cand['start'] = start
            cand['end'] = end
            cand['text'] = new_text

    def _is_garbage(self, cand: Dict[str, Any]) -> bool:
        """Επιστρέφει True αν η πρόβλεψη είναι σκουπίδι."""
        text = cand.get('text', '').strip()
        label = cand.get('label', '')
        source = cand.get('source', '')
        conf = float(cand.get('confidence', 0.0))

        # 1. Empty or Blacklisted
        if not text: return True
        if text.lower() in self.blacklist: return True

        # 2. Tiny Fragments (εκτός αν είναι Lexicon/Regex που τα εμπιστευόμαστε)
        if len(text) < 2: return True # Μονά γράμματα πάντα λάθος
        
        # Αν το Hybrid βρει κάτι < 3 γράμματα (π.χ. "21"), είναι σχεδόν πάντα λάθος.
        if len(text) < 3 and source == 'Hybrid':
            return True

        # 3. Hybrid Low Confidence
        # Αν το Hybrid δεν είναι τουλάχιστον 55% σίγουρο, δεν το θέλουμε.
        if source == 'Hybrid' and conf < 0.55:
            return True

        # 4. Person Safety
        if label == 'PERSON':
            if re.search(r'\d', text): return True
            if not any(c.isupper() for c in text): return True
            
        return False

    def _calculate_score(self, cand: Dict[str, Any]) -> float:
        source = cand.get('source', 'Agent')
        confidence = float(cand.get('confidence', 0.5))
        text = cand.get('text', '')
        
        reliability = self.source_reliability.get(source, 0.70)
        
        # Base Probability
        base_prob = confidence * reliability
        
        # Length Bonus (Logarithmic)
        length_bonus = 0.05 * math.log(len(text) + 1)
        
        # Penalty (Αντί για αφαίρεση, κάνουμε multiplication decay για να μην πάει αρνητικό)
        final_score = base_prob + length_bonus
        
        return round(final_score, 4)