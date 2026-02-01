import sqlite3
import numpy as np
import json

class VectorMemory:
    def __init__(self, db_path="data/annotations.db"):
        self.db_path = db_path
        self.vectors = None
        self.metadata = []
        self.prototypes = {} 
        
        self.rejected_vectors = None
        self.rejected_metadata = []
        
        self.load_memory()

    def load_memory(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Φόρτωση Prototypes (Έτοιμα από τη βάση!)
        cursor.execute("SELECT label, vector FROM prototypes")
        for r in cursor.fetchall():
            self.prototypes[r[0]] = np.frombuffer(r[1], dtype=np.float32)
            
        # 2. Φόρτωση Vectors (Accepted / Golden)
        cursor.execute("""
            SELECT vector, label, text_span, frequency, trigger_text
            FROM annotations 
            WHERE vector IS NOT NULL AND (is_accepted=1 OR is_golden=1)
        """)
        rows = cursor.fetchall()
        
        vec_list = []
        meta_list = []
        
        for r in rows:
            vec_blob, label, text, freq, trig = r
            vector = np.frombuffer(vec_blob, dtype=np.float32)
            
            # DIMENSION CHECK (Critical for V2)
            # We expect 2313 dimensions. If old vector (768), skip or pad?
            # Better to skip to avoid noise.
            if vector.shape[0] != 2313:
                continue
                
            vec_list.append(vector)
            meta_list.append({
                'label': label, 
                'text_span': text, # FIXED: Changed from 'text' to 'text_span' to match consumers
                'frequency': freq,
                'trigger_text': trig
            })
            
        if vec_list:
            self.vectors = np.array(vec_list)
            self.metadata = meta_list
            print(f"🧠 Memory Loaded: {len(self.vectors)} accepted entities, {len(self.prototypes)} prototypes.")
        else:
            print("⚠️ Accepted Memory is empty.")
            
        # 3. Load Rejected Vectors (Negative Memory)
        cursor.execute("""
            SELECT vector, label, text_span 
            FROM annotations 
            WHERE vector IS NOT NULL AND is_rejected=1
        """)
        rows_rej = cursor.fetchall()
        
        rej_vec_list = []
        rej_meta_list = []
        
        for r in rows_rej:
            vec_blob, label, text = r
            vector = np.frombuffer(vec_blob, dtype=np.float32)
            rej_vec_list.append(vector)
            rej_meta_list.append({'label': label, 'text': text})
            
        if rej_vec_list:
            self.rejected_vectors = np.array(rej_vec_list)
            self.rejected_metadata = rej_meta_list
            print(f"🛡️ Negative Memory Loaded: {len(self.rejected_vectors)} rejected items.")
        
        conn.close()

    def find_similar(self, query_vector, k=10, threshold=0.0):
        if self.vectors is None: return []

        # ---------------------------------------------------------
        #  WEIGHTED MULTI-VIEW SEARCH (Version 2.0)
        # ---------------------------------------------------------
        # Instead of one giant cosine similarity on 2313 dims, 
        # we split the vector into Views and average their similarities.
        #
        # Structure: [Prev(768) | Target(768) | Next(768) | Type(~9)]
        #
        # Weights:
        #   - Entity View (Target + Type): 50%
        #   - Context Left (Prev):         25%
        #   - Context Right (Next):        25%
        # ---------------------------------------------------------

        # 1. Define Slices
        DIM_EMB = 768
        
        # Slices
        s_prev   = slice(0, DIM_EMB)
        s_target = slice(DIM_EMB, DIM_EMB*2)
        s_next   = slice(DIM_EMB*2, DIM_EMB*3)
        s_type   = slice(DIM_EMB*3, None) # The rest

        # 2. Extract Query Views
        q_prev   = query_vector[s_prev]
        q_target = query_vector[s_target]
        q_next   = query_vector[s_next]
        q_type   = query_vector[s_type] # Optional, see below

        # Combined Entity View = Target + Type 
        # (We concatenate them to treat them as the "Identity" of the entity)
        q_entity_view = np.concatenate([q_target, q_type])

        # 3. Extract Memory Views (Vectorized)
        # self.vectors shape: (N, 2313)
        m_prev   = self.vectors[:, s_prev]
        m_next   = self.vectors[:, s_next]
        
        # Combined Entity View
        m_target = self.vectors[:, s_target]
        m_type   = self.vectors[:, s_type]
        m_entity_view = np.hstack([m_target, m_type]) # Valid concatenation

        # 4. Compute Similarities for each View
        
        def cosine_sim_batch(matrix, vec):
            # matrix: (N, D), vec: (D,)
            norm_m = np.linalg.norm(matrix, axis=1)
            norm_v = np.linalg.norm(vec)
            
            denom = norm_m * norm_v
            denom[denom == 0] = 1e-9
            
            dot = np.dot(matrix, vec)
            return dot / denom

        sim_entity = cosine_sim_batch(m_entity_view, q_entity_view)
        sim_prev   = cosine_sim_batch(m_prev, q_prev)
        sim_next   = cosine_sim_batch(m_next, q_next)

        # 5. Weighted Aggregation
        # Formula: 0.5 * Entity + 0.25 * Prev + 0.25 * Next
        final_scores = (0.5 * sim_entity) + (0.25 * sim_prev) + (0.25 * sim_next)

        # 6. Retrieve Top-K
        top_k_indices = np.argsort(final_scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            score = float(final_scores[idx])
            if score >= threshold:
                res = self.metadata[idx].copy()
                res['similarity'] = score
                # Debug info
                res['debug_sim_entity'] = float(sim_entity[idx])
                res['debug_sim_ctx'] = float((sim_prev[idx] + sim_next[idx]) / 2)
                results.append(res)
                
        return results

    def get_prototype_similarity(self, query_vector):
        scores = {}
        if not self.prototypes: return scores
        
        norm_q = np.linalg.norm(query_vector)
        if norm_q == 0: return scores
        
        for label, centroid in self.prototypes.items():
            norm_c = np.linalg.norm(centroid)
            sim = np.dot(query_vector, centroid) / (norm_q * norm_c)
            scores[label] = float(sim)
        return scores

    def check_is_rejected(self, query_vector, label, threshold=0.95):
        """
        Check if the query vector is highly similar to a rejected item with the SAME label.
        Returns (True, matching_text) if likely a recurring false positive.
        """
        if self.rejected_vectors is None: return False, None
        
        norm_q = np.linalg.norm(query_vector)
        norms_r = np.linalg.norm(self.rejected_vectors, axis=1)
        
        denom = norms_r * norm_q
        denom[denom == 0] = 1e-9
        
        sims = np.dot(self.rejected_vectors, query_vector) / denom
        
        # Check only potential matches above threshold
        matches_indices = np.where(sims >= threshold)[0]
        
        for idx in matches_indices:
            rej_label = self.rejected_metadata[idx]['label']
            if rej_label == label:
                return True, self.rejected_metadata[idx]['text']
                
        return False, None