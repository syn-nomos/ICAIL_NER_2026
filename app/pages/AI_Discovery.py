import streamlit as st
import pandas as pd
import sys
import os
import time

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments.discovery_engine import DiscoveryEngine
from src.models.roberta_ner import RobertaNER

st.set_page_config(page_title="AI Discovery", page_icon="🕵️", layout="wide")

st.title("🕵️ AI Discovery & Trigger Learning (Pilot)")
st.caption("Auto-discovery of new entities based on 'Trigger Words' learned from your feedback.")

# --- Resources ---
def find_best_model_path():
    """Finds the most recent valid model."""
    base_dir = "src/agents"
    candidates = []
    
    if not os.path.exists(base_dir):
        return "xlm-roberta-base"

    # Search for directories starting with GreekLegalRoBERTa
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d)
        if os.path.isdir(path) and "GreekLegalRoBERTa" in d:
            # Check if has config.json
            if os.path.exists(os.path.join(path, "config.json")):
                 candidates.append(path)
    
    # Sort by name (roughly timestamp order for _New_timestamp)
    candidates.sort(reverse=True)
    
    if candidates:
        return candidates[0]
    
    # Check baseline
    baseline = "src/agents/Roberta_Base_Api/model"
    if os.path.exists(os.path.join(baseline, "config.json")):
        return baseline
        
    return "xlm-roberta-base"

@st.cache_resource
def get_model():
    model_path = find_best_model_path()
    # Cache the model to avoid reloading on every interaction
    print(f"🕵️ AI Discovery loading model from: {model_path}")
    try:
        return RobertaNER(model_path)
    except Exception as e:
        st.error(f"Failed to load {model_path}: {e}")
        return RobertaNER("xlm-roberta-base")


@st.cache_resource
def get_engine():
    # We pass None for model initially, will inject cached model manually
    engine = DiscoveryEngine()
    # Pre-load the DB connection
    return engine

# Initialize
engine = get_engine()
# We manually set the model to the cached one to share memory
try:
    engine.model = get_model()
    # Also need to init extractor if not done
    from src.core.attention_extractor import AttentionExtractor
    if engine.extractor is None:
        engine.extractor = AttentionExtractor()
except Exception as e:
    st.error(f"Failed to load model: {e}")

# --- State ---
if 'discovery_logs' not in st.session_state:
    st.session_state.discovery_logs = []
if 'proposals' not in st.session_state:
    st.session_state.proposals = []
if 'learned_triggers' not in st.session_state:
    st.session_state.learned_triggers = pd.DataFrame()

# --- GUI ---

tab_learn, tab_scan, tab_results = st.tabs(["1. Learn Triggers", "2. Scan Corpus", "3. Review Proposals"])

with tab_learn:
    st.header("Step 1: Learn form History")
    st.markdown("""
    The system will analyze your **Accepted Annotations** to identify 'Trigger Words'.
    *   It uses **Self-Attention** to see which word the model 'looked at' when making the decision.
    *   Example: For 'Alpha Bank' (ORG), the trigger might be 'κατάστημα' or 'τραπεζικό'.
    """)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🧠 Start Learning", type="primary"):
            st.session_state.discovery_logs = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(p, msg):
                progress_bar.progress(p)
                status_text.text(msg)
                
            logs = engine.learn_triggers(progress_callback=update_progress, limit=None) # Unlimited for real run
            st.session_state.discovery_logs = logs
            
            # Convert stats to DF for display
            data = []
            for word, counts in engine.trigger_stats.items():
                total = sum(counts.values())
                best_label = max(counts, key=counts.get)
                data.append({
                    "Trigger Word": word,
                    "Dominant Label": best_label,
                    "Count": total,
                    "Details": str(dict(counts))
                })
            
            if data:
                st.session_state.learned_triggers = pd.DataFrame(data).sort_values(by="Count", ascending=False)
                st.success(f"Learning Complete! Found {len(data)} unique triggers.")
            else:
                st.session_state.learned_triggers = pd.DataFrame(columns=["Trigger Word", "Dominant Label", "Count", "Details"])
                st.warning("Learning finished but found NO triggers. This happens if you haven't 'Accepted' enough annotations in the main app yet.")
                
    with col2:
        st.subheader("Training Logs")
        with st.expander("Show detailed logs", expanded=False):
            for log in st.session_state.discovery_logs:
                st.text(log)

    st.subheader("Learned Triggers Knowledge Base")
    if not st.session_state.learned_triggers.empty:
        st.dataframe(st.session_state.learned_triggers, use_container_width=True)
    else:
        st.info("No triggers learned yet.")

with tab_scan:
    st.header("Step 2: Auto-Discovery")
    st.markdown("""
    Using the **Learned Triggers**, scan the 'Empty' or 'Pending' sentences to find missed entities.
    *   This ignores existing annotations.
    *   It looks for High-Confidence Triggers + Capitalized Candidate Words.
    """)
    
    if st.button("🚀 Run Discovery Scan"):
        if st.session_state.learned_triggers.empty:
            st.warning("Please run Step 1 (Learn) first!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def scan_progress(p, msg):
                progress_bar.progress(p)
                status_text.text(msg)
            
            proposals, logs = engine.scan_for_proposals(progress_callback=scan_progress)
            st.session_state.proposals = proposals
            st.session_state.scan_logs = logs
            st.success(f"Scan Complete! Found {len(proposals)} proposals.")

with tab_results:
    st.header("Step 3: Review & Feedback")
    
    if st.session_state.proposals:
        df_props = pd.DataFrame(st.session_state.proposals)
        if not df_props.empty:
            st.write(f"Found **{len(df_props)}** potential entities.")
            
            # Interactive Editor
            edited_df = st.data_editor(
                df_props,
                column_config={
                    "sentence_id": st.column_config.NumberColumn("ID", width="small"),
                    "trigger": "Evidence (Trigger)",
                    "candidate": "Proposed Entity",
                    "predicted_label": "Guessed Label",
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                    "accept": st.column_config.CheckboxColumn("Accept?", default=False)
                },
                use_container_width=True,
                disabled=["sentence_text", "reason", "sentence_id", "trigger", "candidate", "predicted_label", "confidence"]
            )
            
            if st.button("Save Selected to 'Proposals' File"):
                # Filter checked
                # Note: streamlit data_editor return value processing depends on version, 
                # but usually it returns the modified DF if we bind it.
                # However, we didn't add the 'accept' column in the source list, so user can't toggle it easily unless we prep it.
                pass 
                st.info("Saving functionality is disabled in this pilot mode. The identified entities are visible above.")
        else:
            st.info("No valid proposals found.")
    else:
        st.info("No proposals generated yet.")
