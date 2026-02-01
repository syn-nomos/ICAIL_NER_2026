import streamlit as st
import time
import pandas as pd
from typing import List, Dict, Any
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.database.db_manager import DBManager
from src.core.hybrid_predictor import HybridPredictor
from src.judges.llm_client import LLMJudge
from src.core.council import Council
from src.utils.text_utils import get_entity_color

# Page Config
st.set_page_config(page_title="🤖 Multi-Model AI Council", layout="wide")

def get_sentence_batch(db, limit=10):
    return db.conn.execute("SELECT id, text FROM sentences WHERE status='pending' LIMIT ?", (limit,)).fetchall()

def main():
    st.title("🏛️ AI Council: Multi-Model Annotation")
    st.markdown("""
    **The Three-Tier Architecture:**
    *   **Tier 0 (Local):** RoBERTa + Regex + Logic. (Fast, Base Precision)
    *   **Tier 1 (Scanner):** Google Gemini 1.5 Flash. (High Recall, Discovery)
    *   **Tier 2 (Judge):** DeepSeek V3. (Reasoning & Conflict Resolution)
    """)
    
    # Init DB
    try:
        db = DBManager("data/production_annotations.db")
    except Exception as e:
        st.error(f"DB Error: {e}")
        return

    # --- MODEL LOADING ---
    if 'council' not in st.session_state:
        # Load heavy models only once
        try:
            from src.models.roberta_ner import RobertaNER
            from src.core.augmented_embeddings import AugmentedEmbeddingBuilder
            from src.core.attention_extractor import AttentionExtractor
            # from src.core.vector_memory import VectorMemory # Disabled for now per user
            
            from src.agents.leg_refs_regex_agent import LegRefsRegexAgent
            from src.agents.gpe_regex_agent import GpeRegexAgent
            from src.agents.org_regex_agent import OrgRegexAgent
            from src.agents.public_docs_regex_agent import PublicDocsRegexAgent
            
            with st.spinner("Initializing AI Council (Loading RoBERTa & Connecting to Cloud)..."):
                # Tier 0 Components
                roberta = RobertaNER("data/models/roberta_finetuned_v2")
                # mem = VectorMemory("data/models/vector_memory_v2.pkl") # TAAL Disabled
                mem = None 
                att = AttentionExtractor()
                emb = AugmentedEmbeddingBuilder()
                
                # Regex Agents
                agents = [
                    LegRefsRegexAgent(), GpeRegexAgent(), OrgRegexAgent(), PublicDocsRegexAgent()
                ]
                
                # Initialize Predictor
                # If memory is None, HybridPredictor should handle it gracefully or we fix it.
                # Assuming current HybridPredictor handles None mem (or we patch it).
                # To be safe, we can mock it if needed, but None usually throws AttributeError if accessed.
                # Let's create a dummy memory if needed. 
                # Checking HybridPredictor source... it calls self.memory.search(). 
                # So passing None will crash unless we modify HybridPredictor.
                # Quick fix: Pass a dummy object.
                class DummyMem:
                    def search(self, *args, **kwargs): return []
                    def add(self, *args, **kwargs): pass
                
                mem_safe = DummyMem()

                predictor = HybridPredictor(roberta, mem_safe, att, emb, rule_agents=agents)
                
                # Initialize Judge
                judge = LLMJudge() # Reads secrets for OpenRouter/DeepSeek
                
                # Create Council
                st.session_state.council = Council(predictor, judge) # Scanner (Gemini) init inside Council
                st.success("The Council is in session.")
                
        except Exception as e:
            st.error(f"Failed to assemble the Council: {e}")
            st.stop()

    # --- UI CONTROLS ---
    c1, c2, c3 = st.columns(3)
    batch_size = c1.number_input("Batch Size", 5, 200, 10)
    delay = c2.number_input("Delay (sec)", 0.0, 5.0, 0.2)
    min_confidence = c3.slider("Minimum Confidence", 0.0, 1.0, 0.6)
    
    if st.button("▶️ EXECUTE COUNCIL PROTOCOL", type="primary"):
        stop_btn = st.button("⏹️ ADJOURN")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_area = st.container(height=500)
        
        # Fetch Batch
        sentences = get_sentence_batch(db, batch_size)
        
        if not sentences:
            st.warning("No pending sentences found!")
            return

        total = len(sentences)
        stats = {'PLATINUM': 0, 'JUDGED': 0, 'DISCOVERY': 0, 'LOCAL': 0}
        
        for i, (sid, text) in enumerate(sentences):
            if stop_btn: break
            
            status_text.text(f"Processing Sentence {i+1}/{total} (ID: {sid})...")
            progress_bar.progress((i+1)/total)
            
            # --- CONVENE THE COUNCIL ---
            # This runs Tier 0, Tier 1 and resolves via Tier 2
            try:
                council_decisions = st.session_state.council.convene(text, min_confidence=min_confidence)
            except Exception as e:
                st.error(f"Council Error on ID {sid}: {e}")
                # Don't stop, skip
                continue
            
            # --- COMMIT RESULTS ---
            saved_count = 0
            
            with log_area:
                st.markdown(f"**Sentence {sid}**: _{text[:60]}..._")
            
            for item in council_decisions:
                # We need to generate the vector for memory future-proofing
                # even if TAAL is disabled for *retrieval*, we should *store* vectors for future training.
                blob = item.get('vector_blob')
                if not blob and 'text' in item:
                    try:
                        idx_start = text.find(item['text'])
                        if idx_start != -1:
                            # Access components via the predictor in the council
                            p = st.session_state.council.tier0
                            v = p.emb_builder.build(p.roberta, text, idx_start, idx_start+len(item['text']), item['label'])
                            if v is not None:
                                blob = v.tobytes()
                    except Exception as e:
                        # Don't fail the batch for a vector error
                        print(f"Vector gen error: {e}")
                
                # Insert
                status_tag = item.get('council_status', 'Unknown')
                source_txt = f"Council: {status_tag}"
                
                db.conn.execute("""
                    INSERT INTO annotations 
                    (sentence_id, text_span, label, confidence, vector, source_sentence, trigger_text, is_accepted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """, (
                    sid, 
                    item['text'], 
                    item['label'], 
                    item.get('confidence', 0.95), 
                    blob, 
                    text, 
                    source_txt
                ))
                saved_count += 1
                
                # Update Stats
                if 'PLATINUM' in status_tag: stats['PLATINUM'] += 1
                elif 'JUDGED' in status_tag: stats['JUDGED'] += 1
                elif 'DISCOVERY' in status_tag: stats['DISCOVERY'] += 1
                else: stats['LOCAL'] += 1
                
                # Visual Log
                with log_area:
                    color = "#bbf7d0" if "PLATINUM" in status_tag else "#fef08a" if "JUDGED" in status_tag else "#bfdbfe"
                    st.markdown(f"<span style='background-color:{color}; padding:2px 5px; border-radius:4px;'>{item['label']}</span> **{item['text']}** ({status_tag})", unsafe_allow_html=True)
            
            # Mark sentence as processed
            db.update_sentence_status(sid, 'approved')
            db.conn.commit()
            
            time.sleep(delay)
            
        st.success(f"Session Adjourned. Processed {total} sentences.")
        st.write("📊 **Session Stats:**", stats)

if __name__ == "__main__":
    main()
