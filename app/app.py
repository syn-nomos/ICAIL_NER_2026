import streamlit as st
import pandas as pd
import numpy as np
import re
import html as html_lib
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from concurrent.futures import ThreadPoolExecutor
from src.judges.llm_client import LLMJudge

# --- AI System Imports ---
from src.phase1_hybrid.models.roberta_ner import RobertaNER
from src.phase1_hybrid.core.vector_memory import VectorMemory
from src.phase1_hybrid.core.controller import NerController
from src.phase1_hybrid.core.hybrid_predictor import HybridPredictor

# Agents
from src.phase1_hybrid.agents.leg_refs_regex_agent import LegRefsRegexAgent
from src.phase1_hybrid.agents.leg_refs_lexicon_agent import LegRefsLexiconAgent
from src.phase1_hybrid.agents.public_docs_regex_agent import PublicDocsRegexAgent
from src.phase1_hybrid.agents.public_docs_lexicon_agent import PublicDocsLexiconAgent
# ... (all other agents can be loaded dynamically or here)

from src.utils.text_utils import (
    normalize_text, get_pseudo_stem, get_entity_color, 
    get_entity_border, highlight_sentence, render_tokenized_text,
    adjust_boundaries
)
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

# --- Streamlit Fragment Fallback (For older versions) ---
if not hasattr(st, "fragment"):
    def _fragment_fallback(func):
        return func
    st.fragment = _fragment_fallback

# --- Database Connection Strategy ---
def validate_db_connection(manager):
    """Checks if the DB connection is still alive."""
    try:
        if hasattr(manager, 'check_connection'):
            return manager.check_connection()
        # Fallback for SQLite or older classes
        manager.conn.cursor().execute("SELECT 1")
        return True
    except Exception:
        return False

@st.cache_resource(ttl=3600, validate=validate_db_connection) # Cache for 1 hour, validate connection
def get_db_manager():
    import importlib
    import sys
    
    # Check for DATABASE_URL in Env or Secrets
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        try:
            # Try direct key
            if "DATABASE_URL" in st.secrets:
                db_url = st.secrets["DATABASE_URL"]
            # Try nested 'postgres' section (common in Streamlit docs)
            elif "postgres" in st.secrets and "url" in st.secrets["postgres"]:
                db_url = st.secrets["postgres"]["url"]
            # Try 'supabase' section
            elif "supabase" in st.secrets and "url" in st.secrets["supabase"]:
                db_url = st.secrets["supabase"]["url"]
        except Exception as e:
            print(f"Error reading secrets: {e}")
            pass

    # If DATABASE_URL is set, we use PostgreSQL (Supabase), otherwise SQLite (Local)
    if db_url:
        # Ensure env var is set for compatibility
        os.environ["DATABASE_URL"] = db_url
        
        if 'src.database.db_manager_postgres' in sys.modules:
            importlib.reload(sys.modules['src.database.db_manager_postgres'])
        from src.database.db_manager_postgres import DBManager
        print("🚀 Using PostgreSQL Database (Reloaded)")
        
        try:
            return DBManager(db_url=db_url)
        except Exception as e:
            st.error("### 🔌 Database Connection Failed")
            st.warning(f"""
            **Could not connect to the database.**
            
            Error details: `{e}`
            
            **Troubleshooting for Streamlit Cloud:**
            1. Go to **Manage App** -> **Settings** -> **Secrets**.
            2. Ensure you are using the **Transaction Pooler URL** (Port 6543).
            3. The URL should look like: `postgres://user:pass@...pooler.supabase.com:6543/postgres`
            4. If you have a `[supabase]` section in secrets, it might be overriding `DATABASE_URL` with the wrong port (5432).
            """)
            st.stop()
    else:
        if 'src.database.db_manager' in sys.modules:
            importlib.reload(sys.modules['src.database.db_manager'])
        from src.database.db_manager import DBManager
        print("📂 Using SQLite Database (Reloaded)")
        return DBManager()

db = get_db_manager()

# --- AI System Loader ---
@st.cache_resource
def get_ai_system():
    print("🤖 Loading AI System for Batch Processing...")
    
    # 1. Models
    try:
        roberta = RobertaNER("src/agents/Roberta_Base_Api/model")
        memory = VectorMemory("data/annotations.db")
        # Instantiate Controller WITH Memory & Model Wrapper
        controller = NerController(memory=memory, model_wrapper=roberta)
        hybrid = HybridPredictor(roberta, memory)
    except Exception as e:
        st.error(f"Failed to load AI Models: {e}")
        return None, None, None, None

    # 2. Agents
    agents = [
        LegRefsRegexAgent(), LegRefsLexiconAgent(),
        PublicDocsRegexAgent(), PublicDocsLexiconAgent()
        # Add others if needed... getting heavy though
    ]
    
    return roberta, memory, controller, agents

# --- Jump to Sentence Logic (From Statistics Page) ---
if "target_sentence_id" in st.session_state:
    target_id = st.session_state.pop("target_sentence_id")
    # We need to find the index of this sentence in the current filtered view OR reset filters
    # For simplicity, let's try to find it in the global list first to get the index if no filters were applied
    # But since the main view uses pagination and filters, it's safer to just set a special "Jump Mode"
    # or try to locate it.
    
    # Strategy: Reset filters to ensure visibility, then find index
    # However, resetting filters might be annoying.
    # Let's just try to find the index in the *all sentences* list and set the page accordingly?
    # The app uses `db.get_filtered_sentences(limit=1, offset=current_page)`.
    # So `current_page` is actually the index if limit=1.
    
    # Let's find the index of the target_id in the FULL list of sentences (assuming default sort ID)
    # This is a bit complex because of the filters.
    # Simplest approach: Reset filters to "All" and find the index.
    
    # 1. Reset Filters (Optional, but safer)
    # st.session_state.status_filter = "All" 
    # (We can't easily reset sidebar widgets programmatically without rerun tricks)
    
    # 2. Find Index
    try:
        # Fetch all IDs to find the index
        all_ids_query = "SELECT id FROM sentences ORDER BY id" # Must match default sort
        cursor = db.conn.cursor()
        cursor.execute(all_ids_query)
        all_rows = cursor.fetchall()
        
        # Handle different cursor types
        # Check if it's a dict-like object (Postgres RealDictRow)
        first_row = all_rows[0]
        if isinstance(first_row, dict) or (hasattr(first_row, 'keys') and callable(first_row.keys)):
             all_ids = [r['id'] for r in all_rows]
        else:
             all_ids = [r[0] for r in all_rows]
             
        if target_id in all_ids:
            idx = all_ids.index(target_id)
            st.session_state.current_page = idx
            st.toast(f"Jumped to Sentence #{target_id}")
        else:
            st.error(f"Sentence {target_id} not found in DB.")
            
    except Exception as e:
        st.error(f"Jump failed: {e}")

# --- Configuration ---
st.set_page_config(
    page_title="UAegean Legal NER Annotator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.dialog("📖 Οδηγός Χρήσης & Τεκμηρίωση")
def show_help_modal():
    st.markdown("""
    ## 👋 Καλώς ήρθατε στο Legal NER Annotator
    
    Αυτή η εφαρμογή σχεδιάστηκε για την **ημι-αυτόματη σχολιασμό νομικών κειμένων** (Named Entity Recognition). 
    Συνδυάζει αυτόματες προβλέψεις από μοντέλα (RoBERTa, Regex) με τη δική σας ανθρώπινη κρίση.

    ---

    ### 1️⃣ Ροή Εργασίας (Workflow)
    
    Η διαδικασία είναι απλή:
    1.  **Επιλογή Πρότασης:** Χρησιμοποιήστε τα βελάκια `Previous` / `Next` ή τη μπάρα προόδου για να πλοηγηθείτε.
    2.  **Επισκόπηση Οντοτήτων:** Οι οντότητες εμφανίζονται με χρώματα μέσα στο κείμενο.
        - 🔵 **Μπλε (Pending):** Χρειάζεται έλεγχο.
        - 🟢 **Πράσινο (Accepted):** Έχει επιβεβαιωθεί.
        - 🔴 **Κόκκινο (Rejected):** Έχει απορριφθεί.
    3.  **Ενέργειες:**
        - ✅ **Accept:** Η οντότητα είναι σωστή (κείμενο και κατηγορία).
        - ❌ **Reject:** Η οντότητα είναι λάθος (δεν είναι οντότητα).
        - ✏️ **Edit:** Η οντότητα χρειάζεται διόρθωση (στα όρια ή στην κατηγορία).

    ---

    ### 2️⃣ Εργαλεία Επεξεργασίας (Editing)
    
    Όταν πατάτε **Edit (✏️)**, ανοίγει το πάνελ επεξεργασίας:
    - **Όρια (Boundaries):**
        - `⬅️ Expand Left` / `Shrink Left ➡️`: Αυξομείωση της αρχής της λέξης.
        - `⬅️ Shrink Right` / `Expand Right ➡️`: Αυξομείωση του τέλους της λέξης.
    - **Precision Mode:** Αν τα κουμπιά δεν αρκούν, ανοίξτε το "Precision Mode" για να ορίσετε ακριβείς δείκτες χαρακτήρων (indices).
    - **Label:** Αλλάξτε την κατηγορία αν το μοντέλο έκανε λάθος (π.χ. από ORG σε GPE).

    ---

    ### 3️⃣ Έξυπνα Βοηθήματα
    
    - **🤖 Ask LLM Assistant:**
        - Βρίσκεται κάτω από την πρόταση.
        - Ρωτήστε τον AI βοηθό για αμφιβολίες (π.χ. *"Είναι το 'Δήμος Αθηναίων' GPE ή ORG;"*).
        - Μπορείτε να επιλέξετε αν θα αναλύσει όλη την πρόταση ή συγκεκριμένο τμήμα.
    
    - **🧠 Memory Propagation (Εικονίδιο Εγκεφάλου):**
        - Εμφανίζεται στις "Accepted Entities".
        - **Λειτουργία:** Ψάχνει σε **όλο το κείμενο** για την ίδια ακριβώς φράση (ή πολύ παρόμοια μέσω **Cosine Similarity** στα embeddings) και την μαρκάρει αυτόματα ως Accepted.
        - **Φίλτρα:** Μπορείτε να ορίσετε πόσο αυστηρή θα είναι η αναζήτηση (Similarity Threshold).
        - **Search Mode:**
            - **Adaptive Centroid:** Μαθαίνει από τις επιλογές σας (μέσος όρος διανυσμάτων) για πιο ακριβή αποτελέσματα.
            - **Original Vector:** Χρησιμοποιεί αυστηρά το διάνυσμα της αρχικής οντότητας.
        - *Χρήση:* Ιδανικό για επαναλαμβανόμενες οντότητες (π.χ. "Τράπεζα Πειραιώς").

    - **🔍 Fuzzy Search (Μεγεθυντικός Φακός):**
        - **Λειτουργία:** Ψάχνει για **παρόμοιες** φράσεις στο **ακατέργαστο κείμενο (sentences)**.
        - **Ρυθμίσεις:**
            - **Fuzzy Threshold:** Πόσο "κοντά" πρέπει να είναι η φράση (π.χ. 85%).
            - **Min Length %:** Ελάχιστο ποσοστό μήκους της λέξης που πρέπει να ταιριάζει (π.χ. 80%).
            - **Show already annotated:** Εμφάνιση αποτελεσμάτων που έχουν ήδη σημειωθεί (για έλεγχο διπλότυπων).
        - **Conflict Resolution:** Αν βρεθεί κάτι και το κάνετε Accept, το σύστημα ελέγχει αν πέφτει πάνω σε άλλη υπάρχουσα οντότητα. Αν ναι, η παλιά (λανθασμένη) απορρίπτεται αυτόματα.
        - *Χρήση:* Χρήσιμο για να βρείτε οντότητες που ξέφυγαν τελείως από το μοντέλο ή έχουν μικρά ορθογραφικά λάθη.

    ---

    ### 5️⃣ Φίλτρα & Πλοήγηση
    
    Στην αριστερή στήλη (Sidebar) υπάρχουν φίλτρα για να εστιάσετε σε συγκεκριμένα δεδομένα:
    - **Status:** Δείτε μόνο Pending, Accepted ή Rejected προτάσεις.
    - **Label / Source:** Φιλτράρετε ανά κατηγορία ή μοντέλο προέλευσης.
    - **Confidence:** Δείτε μόνο οντότητες με υψηλή ή χαμηλή βεβαιότητα.
    - **Flagged Only:** Εμφανίστε μόνο προτάσεις που έχετε επισημάνει με σημαία (🚩).
    - **Reset Filters:** Καθαρίζει όλα τα φίλτρα και επιστρέφει στην αρχή.

    ---

    ### 6️⃣ Συντομεύσεις Πληκτρολογίου
    - `A`: Accept (Πρώτη Pending οντότητα)
    - `R`: Reject (Πρώτη Pending οντότητα)
    - `N`: Next Sentence
    - `P`: Previous Sentence

    | **DATE** | Χρονικές εκφράσεις. | *1η Ιανουαρίου 2020, το έτος 2015* |

    #### ⚠️ Προσοχή: GPE vs LOCATION vs FACILITY
    - **GPE:** Όταν η τοποθεσία **"δρα"** ή έχει εξουσία (π.χ. "Η **Ελλάδα** υπέγραψε").
    - **LOCATION:** Όταν είναι απλώς **τόπος** (π.χ. "ταξίδεψε στην **Ελλάδα**").
    - **FACILITY:** Όταν αναφερόμαστε στο **κτίσμα** (π.χ. "μπήκε στο **Μέγαρο Μαξίμου**").

    ---
    
    ### 5️⃣ Συντομεύσεις Πληκτρολογίου
    - **Ctrl + Enter:** Εκτέλεση ενεργειών (σε φόρμες).
    - **R:** Επαναφόρτωση εφαρμογής (Rerun).
    """)

def update_boundary_callback(ann_id, key_prefix, full_text, action):
    """Callback to handle boundary updates safely before widget rendering."""
    # Determine which keys to use based on prefix
    if key_prefix == "fuz":
        start_key = f"fuz_start_{ann_id}"
        end_key = f"fuz_end_{ann_id}"
    elif key_prefix == "mem":
        start_key = f"mem_start_{ann_id}"
        end_key = f"mem_end_{ann_id}"
    else:
        start_key = f"edit_start_{ann_id}"
        end_key = f"edit_end_{ann_id}"

    s = st.session_state.get(start_key, 0)
    e = st.session_state.get(end_key, 0)
    
    ns, ne = adjust_boundaries(full_text, s, e, action)
    
    # Fallbacks
    if action == "expand_left" and ns == s and s > 0: ns -= 1
    elif action == "shrink_left" and ns == s and s < e: ns += 1
    elif action == "shrink_right" and ne == e and e > s: ne -= 1
    elif action == "expand_right" and ne == e and e < len(full_text): ne += 1
        
    st.session_state[start_key] = ns
    st.session_state[end_key] = ne
    
    # Sync widget state if it exists (for Precision Mode widgets)
    # The keys in render_edit_interface are f"{key_prefix}_ns_{ann_id}" and f"{key_prefix}_ne_{ann_id}"
    # Note: ann_id passed here might be a string like "123_0" for fuzzy/mem, or int ID for edit.
    # In render_edit_interface, ann['id'] is used.
    # For fuzzy/mem, the keys are constructed differently in the UI, but they don't use number_input for precision usually.
    # Only render_edit_interface uses number_input with keys f"{key_prefix}_ns_{ann['id']}".
    
    widget_ns_key = f"{key_prefix}_ns_{ann_id}"
    widget_ne_key = f"{key_prefix}_ne_{ann_id}"
    
    if widget_ns_key in st.session_state:
        st.session_state[widget_ns_key] = ns
    if widget_ne_key in st.session_state:
        st.session_state[widget_ne_key] = ne

def expand_title_callback(s_key, e_key, full_text, label_type):
    """Callback to handle title expansion safely before widget rendering."""
    s = st.session_state.get(s_key, 0)
    e = st.session_state.get(e_key, 0)
    
    if label_type == 'leg-refs':
        from src.utils.text_utils import find_quote_span
        ns, ne = find_quote_span(full_text, s, e)
    elif label_type == 'public-docs':
        from src.utils.public_docs_utils import find_public_docs_title_span
        ns, ne = find_public_docs_title_span(full_text, s, e)
    else:
        ns, ne = s, e
        
    if (ns, ne) != (s, e):
        st.session_state[s_key] = ns
        st.session_state[e_key] = ne
        st.toast(f"Expanded: {s}-{e} ➡️ {ns}-{ne}")
        # Force a rerun to ensure the UI updates immediately, especially within fragments
        st.rerun()
    else:
        st.toast("No expansion found.")
    
    # Sync widgets only for edit mode (others don't have number inputs)
    if key_prefix not in ["fuz", "mem"]:
        st.session_state[f"{key_prefix}_ns_{ann_id}"] = ns
        st.session_state[f"{key_prefix}_ne_{ann_id}"] = ne
    
    # Optional: Toast (might not show if rerun happens immediately, but callbacks usually support it)
    # st.toast(f"Adjusted: {s}:{e} -> {ns}:{ne}")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f5f7f9; }
    .sentence-box {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        font-size: 1.1em;
        line-height: 1.6;
        border-left: 5px solid #4e73df;
    }
    .annotation-card {
        background-color: white;
        border: 1px solid #e3e6f0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .annotation-card:hover {
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .accepted-tag {
        background-color: #1cc88a;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    .pending-tag {
        background-color: #f6c23e;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .focus-mode-container {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
    }
    .entity-highlight {
        border-radius: 4px;
        padding: 2px 4px;
        font-weight: 500;
        border-bottom: 2px solid;
    }
    .entity-badge {
        font-size: 0.75em;
        font-weight: bold;
        margin-right: 4px;
        padding: 1px 5px;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.7);
        color: #333;
        vertical-align: middle;
    }
    /* Indexed Token Visualization */
    .token-box {
        display: inline-block;
        margin: 3px;
        padding: 2px 6px;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background: #fff;
        line-height: 1.2;
    }
    .idx-start {
        color: #d32f2f;
        font-size: 0.65em;
        font-weight: bold;
        vertical-align: top;
        margin-right: 2px;
    }
    .idx-end {
        color: #1976d2;
        font-size: 0.65em;
        font-weight: bold;
        vertical-align: bottom;
        margin-left: 2px;
    }
    .token-container {
        display: inline-block;
        margin: 0 1px;
        position: relative;
        padding-top: 14px; /* Space for numbers above */
        vertical-align: bottom;
    }
    .token-word {
        display: block;
        padding: 1px 3px;
        background: transparent;
        border-bottom: 1px solid #e0e0e0;
        font-family: monospace;
        font-size: 1.0em;
        color: #444;
        line-height: 1.2;
    }
    .token-word:hover {
        background-color: #f1f1f1;
        border-bottom-color: #bbb;
    }
    .idx-badge-start {
        position: absolute;
        top: 0;
        left: 0;
        color: #e53935; /* Red */
        font-size: 0.65em;
        font-weight: bold;
        line-height: 1;
    }
    .idx-badge-end {
        position: absolute;
        top: 0;
        right: 0;
        color: #1e88e5; /* Blue */
        font-size: 0.65em;
        font-weight: bold;
        line-height: 1;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def render_edit_interface(ann, full_text, all_labels, key_prefix):
    """
    Renders a sophisticated edit interface with boundary adjustment controls.
    """

    # --- LEG-REFS: Ειδικό κουμπί για επέκταση σε τίτλο ---
    label_val = str(ann.get('label', '')).strip().lower()
    s = st.session_state.get(f"edit_start_{ann['id']}", ann.get('start_char', 0))
    e = st.session_state.get(f"edit_end_{ann['id']}", ann.get('end_char', 0))
    if label_val == 'leg-refs':
        from src.utils.text_utils import find_quote_span
        q_start, q_end = find_quote_span(full_text, s, e)
        if st.button("Επέκταση σε τίτλο (auto)", key=f"expand_title_{ann['id']}"):
            if (q_start, q_end) != (s, e):
                st.session_state[f"edit_start_{ann['id']}"] = q_start
                st.session_state[f"edit_end_{ann['id']}"] = q_end
                # Sync widget state to prevent revert
                st.session_state[f"{key_prefix}_ns_{ann['id']}"] = q_start
                st.session_state[f"{key_prefix}_ne_{ann['id']}"] = q_end
                st.rerun()
            else:
                st.info("Δεν βρέθηκε τίτλος για επέκταση.")
    elif label_val == 'public-docs':
        from src.utils.public_docs_utils import find_public_docs_title_span
        t_start, t_end = find_public_docs_title_span(full_text, s, e)
        if st.button("Επέκταση σε τίτλο (auto)", key=f"expand_title_pubdocs_{ann['id']}"):
            if (t_start, t_end) != (s, e):
                st.session_state[f"edit_start_{ann['id']}"] = t_start
                st.session_state[f"edit_end_{ann['id']}"] = t_end
                # Sync widget state to prevent revert
                st.session_state[f"{key_prefix}_ns_{ann['id']}"] = t_start
                st.session_state[f"{key_prefix}_ne_{ann['id']}"] = t_end
                st.rerun()
            else:
                st.info("Δεν βρέθηκε τίτλος για επέκταση.")
    else:
        st.caption(f"[DEBUG] Label: '{ann.get('label')}' (κουμπί LEG-REFS/PUBLIC-DOCS δεν εμφανίζεται)")
    # Init state for this edit session if not exists
    if f"edit_start_{ann['id']}" not in st.session_state:
        st.session_state[f"edit_start_{ann['id']}"] = ann['start_char']
        st.session_state[f"edit_end_{ann['id']}"] = ann['end_char']

    s = st.session_state.get(f"edit_start_{ann['id']}", ann.get('start_char', 0))
    e = st.session_state.get(f"edit_end_{ann['id']}", ann.get('end_char', 0))
    
    # Preview Context
    # Show context: ... prev [ TARGET ] next ...
    prev_ctx = full_text[max(0, s-40):s]
    sel_ctx = full_text[s:e]
    next_ctx = full_text[e:min(len(full_text), e+40)]
    
    st.markdown(f"""
    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:10px; font-family:monospace;">
        <span style="color:#6c757d">...{prev_ctx}</span>
        <span style="background-color:#fff3cd; padding:2px 5px; border-radius:3px; font-weight:bold; border:1px solid #ffeeba;">{sel_ctx}</span>
        <span style="color:#6c757d">{next_ctx}...</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Indexed View for Precision
    with st.expander("📏 Precision Mode (Indices)", expanded=False):
        st.caption("Use the Start/End indices shown on the words to adjust boundaries.")
        
        # Tokenize for display
        st.markdown(render_tokenized_text(full_text), unsafe_allow_html=True)
        
        ic1, ic2 = st.columns(2)
        with ic1:
            new_s = st.number_input("Start Index", min_value=0, max_value=len(full_text), value=s, key=f"{key_prefix}_ns_{ann['id']}")
        with ic2:
            new_e = st.number_input("End Index", min_value=0, max_value=len(full_text), value=e, key=f"{key_prefix}_ne_{ann['id']}")
            
        if new_s != s or new_e != e:
            st.session_state[f"edit_start_{ann['id']}"] = new_s
            st.session_state[f"edit_end_{ann['id']}"] = new_e
            st.rerun()

    # Boundary Controls
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.button("⬅️ Expand Left", key=f"{key_prefix}_el_{ann['id']}", 
                  on_click=update_boundary_callback, 
                  args=(ann['id'], key_prefix, full_text, "expand_left"))
            
    with c2:
        st.button("Shrink Left ➡️", key=f"{key_prefix}_sl_{ann['id']}",
                  on_click=update_boundary_callback, 
                  args=(ann['id'], key_prefix, full_text, "shrink_left"))

    with c3:
        st.button("⬅️ Shrink Right", key=f"{key_prefix}_sr_{ann['id']}",
                  on_click=update_boundary_callback, 
                  args=(ann['id'], key_prefix, full_text, "shrink_right"))

    with c4:
        st.button("Expand Right ➡️", key=f"{key_prefix}_er_{ann['id']}",
                  on_click=update_boundary_callback, 
                  args=(ann['id'], key_prefix, full_text, "expand_right"))

    # Label Select
    new_label = st.selectbox("Label", all_labels, index=all_labels.index(ann['label']) if ann['label'] in all_labels else 0, key=f"{key_prefix}_label_{ann['id']}")
    
    # Save / Cancel
    sc1, sc2 = st.columns([1, 1])
    
    # Overlap check: only with non-rejected annotations, exclude self
    s_new = st.session_state[f"edit_start_{ann['id']}"]
    e_new = st.session_state[f"edit_end_{ann['id']}"]
    
    existing_anns = db.get_annotations_for_sentence(ann['sentence_id'])
    non_rejected = [a for a in existing_anns if not a.get('is_rejected', 0) and a['id'] != ann['id']]
    overlapping = [a for a in non_rejected if (s_new < a['end_char']) and (e_new > a['start_char'])]
    
    do_save = False
    if not overlapping:
        if sc1.button("💾 Save & Accept", key=f"{key_prefix}_save_{ann['id']}", type="primary"):
            do_save = True
    else:
        st.warning(f"⚠️ Υπάρχει overlap με {len(overlapping)} accepted/pending annotation(s). Θα γίνουν reject αν συνεχίσετε.")
        if st.button("Συνέχεια και reject overlaps", key=f"confirm_overlap_btn_{ann['id']}", type="primary"):
            do_save = True

    if do_save:
        st.toast("Saving changes...") # Debug toast
        final_text = full_text[s_new:e_new]
        # Change Detection
        s_old, e_old = ann['start_char'], ann['end_char']
        l_old = ann['label']

        boundaries_changed = (s_new != s_old) or (e_new != e_old)
        label_changed = (new_label != l_old)

        # Determine Resolution Type
        if boundaries_changed and label_changed:
            res_type = "accepted_modified_both"
        elif boundaries_changed:
            res_type = "accepted_modified_boundaries"
        elif label_changed:
            res_type = "accepted_modified_label"
        else:
            res_type = "accepted_verified"

        db.execute_query(
            "UPDATE annotations SET text_span=?, label=?, start_char=?, end_char=?, manual_edit_type=?, is_accepted=?, created_at=CURRENT_TIMESTAMP WHERE id=?", 
            (final_text, new_label, s_new, e_new, res_type, True, ann['id'])
        )

        # Update the annotation dict in-place so that the new boundaries are reflected immediately
        ann['text_span'] = final_text
        ann['label'] = new_label
        ann['start_char'] = s_new
        ann['end_char'] = e_new

        # Reject overlapping annotations (exclude self)
        db.reject_overlapping_annotations(ann['sentence_id'], s_new, e_new, exclude_id=ann['id'])

        # Update centroid if boundaries didn't change (vector still valid for this span)
        if not boundaries_changed and ann.get('vector'):
            db.update_entity_centroid(final_text, new_label, ann['vector'])

        st.session_state[f"editing_{ann['id']}"] = False
        # Cleanup
        if f"edit_start_{ann['id']}" in st.session_state: del st.session_state[f"edit_start_{ann['id']}"]
        if f"edit_end_{ann['id']}" in st.session_state: del st.session_state[f"edit_end_{ann['id']}"]
        if f"confirm_overlap_{ann['id']}" in st.session_state: del st.session_state[f"confirm_overlap_{ann['id']}"]
        
        st.toast(f"Annotation saved as {res_type}!")
        st.rerun()
        
    if sc2.button("Cancel", key=f"{key_prefix}_cancel_{ann['id']}"):
        st.session_state[f"editing_{ann['id']}"] = False
        del st.session_state[f"edit_start_{ann['id']}"]
        del st.session_state[f"edit_end_{ann['id']}"]
        st.rerun()

# --- Monkey Patch for Hot Reloading ---
# Ensure execute_query exists even if the cached object is stale
if not hasattr(db, 'execute_query'):
    def _execute_query(query, params=()):
        cursor = db.conn.cursor()
        cursor.execute(query, params)
        db.conn.commit()
        return cursor
    db.execute_query = _execute_query

if not hasattr(db, 'get_confidence_range'):
    def _get_confidence_range():
        cursor = db.conn.cursor()
        cursor.execute("SELECT MIN(confidence) as min_c, MAX(confidence) as max_c FROM annotations")
        row = cursor.fetchone()
        if row:
            # Handle both dict (Postgres) and tuple/row (SQLite)
            try:
                min_c = row['min_c']
                max_c = row['max_c']
            except (KeyError, TypeError):
                min_c = row[0]
                max_c = row[1]
                
            if min_c is not None and max_c is not None:
                return min_c, max_c
        return 0.0, 1.0
    db.get_confidence_range = _get_confidence_range

if not hasattr(db, 'find_similar_sentences'):
    # This monkey patch is a fallback. The actual method is now in DBManager.
    # But if the object is stale, we define it here too.
    def _find_similar_sentences(text_span, limit=20):
        # Smart Fuzzy Match Logic (Same as in DBManager)
        cursor = db.conn.cursor()
        
        # 1. Exact Substring
        search_term = f"%{text_span.lower()}%"
        # Detect DB Type for placeholder
        ph = "%s" if "psycopg2" in str(type(db.conn)) else "?"
        
        query = f"""
            SELECT id, text 
            FROM sentences 
            WHERE status = 'pending' 
            AND lower(text) LIKE {ph}
            LIMIT {ph}
        """
        cursor.execute(query, (search_term, limit))
        results = [dict(row) for row in cursor.fetchall()]
        
        # 2. Word Intersection
        if len(results) < limit:
            words = [w for w in text_span.lower().split() if len(w) > 2]
            if len(words) > 1:
                conditions = []
                params = []
                for w in words:
                    conditions.append(f"lower(text) LIKE {ph}")
                    params.append(f"%{w}%")
                
                found_ids = [r['id'] for r in results]
                if found_ids:
                    placeholders = ','.join([ph] * len(found_ids))
                    exclude_clause = f"AND id NOT IN ({placeholders})"
                    params.extend(found_ids)
                else:
                    exclude_clause = ""
                
                params.append(limit - len(results))
                
                query_fuzzy = f"""
                    SELECT id, text 
                    FROM sentences 
                    WHERE status = 'pending' 
                    AND {' AND '.join(conditions)}
                    {exclude_clause}
                    LIMIT {ph}
                """
                cursor.execute(query_fuzzy, tuple(params))
                results.extend([dict(row) for row in cursor.fetchall()])
                
        return results
    db.find_similar_sentences = _find_similar_sentences

# Patch removed to use strict implementation from DBManager
# db.reject_overlapping_annotations is now handled natively in src/database/db_manager.py

# --- Custom CSS for Full Screen Spinner ---
st.markdown("""
<style>
/* Full screen spinner hack to block interaction during loading */
div[data-testid="stSpinner"] {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(255, 255, 255, 0.9);
    z-index: 999999;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    backdrop-filter: blur(5px);
}

/* Custom Spinner Animation */
div[data-testid="stSpinner"] > div {
    border-color: #4CAF50 transparent #4CAF50 transparent !important;
    width: 80px !important;
    height: 80px !important;
    border-width: 6px !important;
}

/* Add a text label below the spinner */
div[data-testid="stSpinner"]::after {
    content: "Processing... Please wait.";
    margin-top: 20px;
    font-size: 1.2rem;
    font-weight: bold;
    color: #333;
    font-family: "Source Sans Pro", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if 'current_page' not in st.session_state: st.session_state.current_page = 0
if 'focus_mode' not in st.session_state: st.session_state.focus_mode = False
if 'focus_index' not in st.session_state: st.session_state.focus_index = 0

# --- Sidebar: Filters ---
with st.sidebar:
    if st.button("📖 Οδηγίες Χρήσης (Help)", type="primary", use_container_width=True):
        show_help_modal()
    
    st.header("🔍 Filters")
    
    # Reset filters on dataset_split change
    def on_split_change():
        # Reset Status to All (safe default)
        st.session_state.status_filter_selectbox = "All"
        st.query_params["status"] = "All"
        # Clear multiselects and reset page
        st.session_state.label_filter_multiselect = []
        st.session_state.agent_filter_multiselect = []
        st.session_state.current_page = 0
        
    # 0. Dataset Split Filter
    dataset_split = st.selectbox(
        "📂 Dataset Split", 
        ["train", "test", "validation", "user_input", "test_v2"], 
        index=4, # Default to test_v2
        key="dataset_split_filter",
        on_change=on_split_change
    )

    # --- ACTIVE LEARNING & BATCHING ---
    st.divider()
    st.header("🤖 Active Learning")
    
    with st.expander("🚀 Batch Prediction", expanded=False):
        batch_size = st.select_slider("Size", options=[10, 20, 50, 100], value=10)
        
        if st.button(f"🔮 Predict Next {batch_size} Pending"):
            roberta, memory, controller, agents = get_ai_system()
            
            if controller:
                # 1. Fetch Pending Sentences (No Annotations)
                # Logic: Sentences in current split with 0 annotations OR status='pending' and 0 candidates
                # To be safe, just get sentences with status='pending'
                pending_sents = db.fetch_pending_sentences(dataset_split, limit=batch_size)
                
                if not pending_sents:
                    st.warning("No pending sentences found!")
                else:
                    progress_bar = st.progress(0)
                    for i, sent in enumerate(pending_sents):
                        text = sent['text']
                        sid = sent['id']
                        
                        # Check if already has annotations (skip if so?)
                        # user might want to re-run
                        
                        # 2. Hybrid Prediction
                        candidates = []
                        # RoBERTa
                        try:
                            # Use wrapping function
                            preds = roberta.predict(text) 
                            # Need to enrich with source 'RoBERTa'
                            for p in preds: p['source'] = 'RoBERTa'
                            candidates.extend(preds)
                        except Exception as e:
                            print(e)
                            
                        # Agents
                        for agent in agents:
                            try:
                                candidates.extend(agent.predict(text))
                            except: pass
                            
                        # 3. Resolve
                        final_entities = controller.resolve(candidates, text)
                        
                        # 4. Save
                        if final_entities:
                            # Clear old annotations for this sentence?
                            # db.clear_annotations(sid) # Maybe safer not to delete if we are unsure
                            
                            # Enrich Vectors
                            final_entities = roberta.enrich_spans_with_vectors(text, final_entities)
                            
                            for ent in final_entities:
                                vec_blob = None
                                if ent.get('vector') is not None:
                                    vec_blob = np.array(ent['vector'], dtype=np.float32).tobytes()
                                
                                db.insert_candidate(
                                    sentence_id=sid,
                                    text=ent['text'],
                                    label=ent['label'],
                                    start=ent['start'],
                                    end=ent['end'],
                                    vector=vec_blob,
                                    confidence=float(ent.get('final_score', 0.0)),
                                    source=ent.get('source', 'Hybrid')
                                )
                                
                        # Update status to 'prediction_ready' so we know it has been processed
                        db.update_sentence_status(sid, 'prediction_ready')
                        
                        progress_bar.progress((i + 1) / batch_size)
                    
                    st.success(f"✅ Processed {len(pending_sents)} sentences!")
                    st.rerun()

    if st.button("🧠 Update Memory & Lexicons (Retrain)", help="Reloads Vector Memory and updates Regex Lexicons"):
        roberta, memory, controller, agents = get_ai_system()
        
        # 1. Update Vectors
        if memory:
            memory.load_memory() # Reload from DB
            st.toast("Vectors Updated! 🧠")
            
        # 2. Update Lexicons (Files)
        try:
            from src.scripts.update_lexicons_from_db import update_lexicons
            added_count = update_lexicons()
            if added_count > 0:
                st.toast(f"Lexicons Enhanced: +{added_count} terms! 📚")
            else:
                st.toast("Lexicons are up to date.")
        except Exception as e:
            st.error(f"Lexicon Update Failed: {e}")
            
    st.divider()

    # 1. Initialize Default State (for Pre-Widget access)
    defaults = {
        'status_filter_selectbox': "All",
        'label_filter_multiselect': [],
        'agent_filter_multiselect': [],
        'flagged_filter_checkbox': False,
        'min_conf_slider': 0.0,
        'max_conf_slider': 1.0,
        'ann_count_type_select': 'Total (Active)',
        'ann_count_slider': (0, 50)
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # 2. Build Filter Dict (Using Session State Directly)
    filters = {
        'dataset_split': dataset_split, # Widget already set this
        'status': st.session_state.status_filter_selectbox if st.session_state.status_filter_selectbox != "All" else None,
        'label': st.session_state.label_filter_multiselect,
        'source_agent': st.session_state.agent_filter_multiselect,
        'confidence_min': st.session_state.min_conf_slider,
        'confidence_max': st.session_state.max_conf_slider,
        'flagged_only': st.session_state.flagged_filter_checkbox,
        'min_annotations': st.session_state.ann_count_slider[0],
        'max_annotations': st.session_state.ann_count_slider[1],
        'annotation_count_type': st.session_state.ann_count_type_select
    }
    
    # 3. Calculate Total Count Early
    total_count = db.get_total_filtered_count(filters)

    # --- 4. Render Navigation Controls (Sticky Top) ---
    st.markdown("### 🧭 Navigation")
    
    # Auto-adjust page if out of bounds
    if st.session_state.current_page >= total_count and total_count > 0:
        st.session_state.current_page = max(0, total_count - 1)
        st.rerun()

    col_prev, col_num, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("⬅️", help="Previous Sentence"):
            st.session_state.current_page = max(0, st.session_state.current_page - 1)
            st.rerun()
            
    with col_next:
        if st.button("➡️", help="Next Sentence"):
            if (st.session_state.current_page + 1) * 1 < total_count: 
                st.session_state.current_page += 1
                st.rerun()

    with col_num:
         # Jump to page logic
        if "page_input" not in st.session_state: st.session_state.page_input = 1
        if st.session_state.current_page + 1 != st.session_state.page_input:
            st.session_state.page_input = st.session_state.current_page + 1
        
        def on_page_change_callback():
            st.session_state.current_page = st.session_state.page_input - 1

        st.number_input(
            "Go to #",
            min_value=1,
            max_value=total_count if total_count > 0 else 1,
            key="page_input",
            on_change=on_page_change_callback,
            label_visibility="collapsed"
        )
    
    st.caption(f"Sentence **{st.session_state.current_page + 1}** of **{total_count}**")
    st.divider()

    # Clear Filters Button
    if st.button("❌ Clear All Filters", use_container_width=True):
        on_split_change() # Reuse logic
        st.rerun()

    # 5. Render Filter Widgets (Now below navigation)
    st.subheader("Advanced Filters")
    
    # 5.1 Status Filter
    # Fetch real counts from DB to show in dropdown
    if hasattr(db, 'get_status_counts'):
        # Pass the selected dataset split to get accurate counts
        status_counts = db.get_status_counts(dataset_split)
    else:
        status_counts = {}
        
    # Build stable options (keys only)
    base_statuses = ["pending", "completed"]
    # Sort extra statuses to ensure list stability
    extra_statuses = sorted([s for s in status_counts.keys() if s not in base_statuses])
    status_keys = ["All"] + base_statuses + extra_statuses
    
    def format_status_func(option):
        if option == "All":
            return "All"
        count = status_counts.get(option, 0)
        return f"{option} ({count})"

    # Ensure session state is initialized for this widget to prevent reset
    # Also sync with Query Params URL for deeper persistence (fixes Cloud refresh issues)
    
    # 1. Aggressive Sync from URL (Source of Truth)
    qp = st.query_params.get("status")
    if isinstance(qp, list): qp = qp[0]
    
    if qp and qp in status_keys and st.session_state.status_filter_selectbox != qp:
        st.session_state.status_filter_selectbox = qp

    # 2. Callback to Update URL
    def on_status_change():
        val = st.session_state.status_filter_selectbox
        st.query_params["status"] = val
        # Also reset page when filter changes
        st.session_state.current_page = 0

    # 3. Widget
    st.selectbox(
        "Sentence Status", 
        status_keys, 
        format_func=format_status_func,
        key="status_filter_selectbox",
        on_change=on_status_change
    )
    
    # 4. Sync URL if missing (First Load)
    if "status" not in st.query_params:
        st.query_params["status"] = st.session_state.status_filter_selectbox
    
    # Parse selection (Not needed for filtering as we used session state above, but good for local logic if any)
    
    # 5.2 Label Filter
    @st.cache_data(ttl=600)
    def get_cached_unique_values(column):
        return db.get_unique_values(column)

    all_labels = get_cached_unique_values("label")
    st.multiselect("Filter by Label", all_labels, key="label_filter_multiselect")
    
    # 5.3 Source Agent Filter
    # Extract unique base agents (e.g. "RoBERTa" from "RoBERTa_ORG")
    all_sources_raw = get_cached_unique_values("source_agent")
    unique_agents = set()
    for s in all_sources_raw:
        if s: unique_agents.add(s.split('_')[0])
    st.multiselect("Filter by Source", sorted(list(unique_agents)), key="agent_filter_multiselect")
    
    # 5.4 Confidence Filter
    @st.cache_data(ttl=600)
    def get_cached_confidence_range():
        return db.get_confidence_range()

    db_min_conf, db_max_conf = get_cached_confidence_range()
    # Ensure we have valid floats
    db_min_conf = float(db_min_conf) if db_min_conf is not None else 0.0
    db_max_conf = float(db_max_conf) if db_max_conf is not None else 1.0
    
    st.caption(f"Confidence Range (DB): {db_min_conf:.2f} - {db_max_conf:.2f}")
    
    # Determine slider limits
    slider_max = max(1.0, db_max_conf)
    
    st.slider("Max Confidence", 0.0, slider_max, slider_max, 0.01, help="Filter out annotations with confidence higher than this.", key="max_conf_slider")
    st.slider("Min Confidence", 0.0, slider_max, 0.0, 0.01, help="Filter out annotations with confidence lower than this.", key="min_conf_slider")
    
    # 5.5 Flagged / Commented Filter
    st.checkbox("🚩 Show Flagged Only", value=False, help="Show only sentences that are flagged.", key="flagged_filter_checkbox")

    # 5.6 Annotation Count Filter
    st.subheader("Annotation Count")
    st.selectbox("Count Type", ["Total (Active)", "Pending", "Accepted", "Rejected"], index=0, key="ann_count_type_select")
    
    # Determine max range dynamically if possible, or set a reasonable default
    max_anns_limit = 50
    st.slider("Count Range", 0, max_anns_limit, (0, max_anns_limit), key="ann_count_slider")
    
    # --- Filter Change Detection (Simple Hash Logic moved to top effect due to rerun loop) ---
    # We rely on widgets updating session state to trigger re-run.
    # The 'filters' dict at the top captures the NEW state immediately on rerun.
    # So we don't need manual hash check unless we want to reset page on specific changes,
    # which we already handle via callbacks or logic.
    
    # Logic to reset page if filters changed significantly (e.g. labels)
    # This is tricky because calculate `total_count` happened before we knew if filters changed compared to last run?
    # Actually, if we use session state at top, we compare current session state to... previous?
    # Let's keep it simple: If total_count changes drastically or becomes 0, pagination handles it.
    # If the user changes a filter, they usually want to start from page 0.
    # We can add `on_change=reset_page_callback` to widgets.
    
    def reset_page_callback():
        st.session_state.current_page = 0
    
    # Note: We can't attach callbacks to already rendered widgets easily without refactoring the calls above.
    # But `on_status_change` already resets page.
    # For others, we can rely on manual "Go to 1" or just add callbacks.
    # For now, the "Auto-adjust page" logic (line 890 in original) handles out-of-bounds.


# --- Main Content ---

# Fetch ONE sentence based on current page
sentences = db.get_filtered_sentences(filters, limit=1, offset=st.session_state.current_page)

if not sentences:
    if total_count == 0:
        st.warning("No sentences found matching your filters.")
        if st.button("🔄 Reset Filters & Start Over"):
            # Reset widgets by clearing session state keys
            keys_to_reset = [
                "status_filter_selectbox",
                "label_filter_multiselect",
                "agent_filter_multiselect",
                "max_conf_slider",
                "min_conf_slider",
                "flagged_filter_checkbox",
                "ann_count_type_select",
                "ann_count_slider"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Reset page
            st.session_state.current_page = 0
            st.rerun()
        st.stop()
    else:
        # Should be handled by auto-adjust above, but just in case
        st.session_state.current_page = 0
        st.rerun()

current_sentence = sentences[0]

# --- Sentence Metadata (Flag & Comments) ---
# Moved to main area in an expander as requested
with st.expander("📝 Sentence Notes & Flags", expanded=False):
    c_meta1, c_meta2 = st.columns([1, 3])
    
    # Use .get() to avoid KeyError if DB migration hasn't fully propagated in cache
    val_flag = bool(current_sentence.get('is_flagged', 0))
    val_comment = current_sentence.get('comments', "")
    if val_comment is None: val_comment = ""
    
    with c_meta1:
        is_flagged = st.checkbox("🚩 Flag for Review", value=val_flag, key=f"flag_{current_sentence['id']}")
        if st.button("Save Notes", key=f"save_meta_{current_sentence['id']}"):
            # We need to read the comment from session state because text_area might not have updated the variable 'comments' yet if we didn't use on_change
            # But here we use the return value of text_area which is fine if button is clicked after typing (and losing focus)
            # Actually, to be safe, we use the key.
            final_comment = st.session_state.get(f"comment_{current_sentence['id']}", val_comment)
            
            db.execute_query("UPDATE sentences SET is_flagged=?, comments=? WHERE id=?", (is_flagged, final_comment, current_sentence['id']))
            st.toast("Notes saved!")
            st.rerun()
            
    with c_meta2:
        comments = st.text_area("Comments", value=val_comment, height=100, key=f"comment_{current_sentence['id']}")

@st.fragment
def render_main_annotation_interface(current_sentence, all_labels):
    annotations = db.get_annotations_for_sentence(current_sentence['id'])

    # Separate Annotations
    # Accepted: is_accepted=1
    # Pending: is_accepted=0 AND is_rejected=0 (or NULL)
    # Rejected: is_rejected=1
    accepted_anns = [a for a in annotations if a['is_accepted']]
    pending_anns = [a for a in annotations if not a['is_accepted'] and not a.get('is_rejected', 0)]
    rejected_anns = [a for a in annotations if a.get('is_rejected', 0)]

    # Assign display index to pending annotations for visualization
    for idx, ann in enumerate(pending_anns):
        ann['display_index'] = idx + 1

    # Combine for highlighting (Only Accepted and Pending are highlighted usually, but let's keep it consistent)
    all_anns_for_display = accepted_anns + pending_anns

    # --- 1. Sentence Display ---
    highlighted_text = highlight_sentence(current_sentence['text'], all_anns_for_display)
    st.markdown(f"<div class='sentence-box'>{highlighted_text}</div>", unsafe_allow_html=True)

    # --- 2. Pending Annotations (Interactive) ---
    st.subheader("⏳ Pending Review")

    if not pending_anns:
        st.success("No pending annotations for this sentence! 🎉")
    else:
        # Toggle Focus Mode
        col_toggle, _ = st.columns([0.2, 0.8])
        if col_toggle.button("🔍 Focus Mode" if not st.session_state.focus_mode else "📋 List View"):
            st.session_state.focus_mode = not st.session_state.focus_mode
            st.rerun()

        if st.session_state.focus_mode:
            # --- FOCUS MODE (One by One) ---
            if st.session_state.focus_index >= len(pending_anns):
                st.session_state.focus_index = 0
                
            ann = pending_anns[st.session_state.focus_index]
            
            with st.container():
                st.markdown("<div class='focus-mode-container'>", unsafe_allow_html=True)
                
                # Header
                st.markdown(f"## {ann['text_span']}")
                st.markdown(f"**Label:** `{ann['label']}` | **Conf:** `{ann['confidence']:.2f}` | **Source:** `{ann['source_agent']}`")
                
                st.divider()
                
                # Actions
                c1, c2, c3 = st.columns(3)
                
                confirm_key_focus = f"confirm_accept_focus_{ann['id']}"
                
                if st.session_state.get(confirm_key_focus, False):
                    with c1:
                        st.warning("⚠️ Overlap!")
                        if st.button("Confirm", key=f"conf_focus_{ann['id']}", use_container_width=True):
                            db.execute_query("UPDATE annotations SET is_accepted=?, manual_edit_type=?, created_at=CURRENT_TIMESTAMP WHERE id=?", (True, 'manual_accept', ann['id']))
                            db.reject_overlapping_annotations(ann['sentence_id'], ann['start_char'], ann['end_char'], exclude_id=ann['id'])
                            st.toast(f"Accepted: {ann['text_span']}")
                            del st.session_state[confirm_key_focus]
                            if st.session_state.focus_index < len(pending_anns) - 1:
                                st.session_state.focus_index += 1
                            st.rerun()
                        if st.button("Cancel", key=f"canc_focus_{ann['id']}", use_container_width=True):
                            del st.session_state[confirm_key_focus]
                            st.rerun()
                else:
                    if c1.button("✅ Accept", key=f"accept_focus_{ann['id']}", use_container_width=True):
                        if db.check_overlapping_annotations(ann['sentence_id'], ann['start_char'], ann['end_char'], exclude_id=ann['id']):
                            st.session_state[confirm_key_focus] = True
                            st.rerun()
                        else:
                            db.execute_query("UPDATE annotations SET is_accepted=?, manual_edit_type=?, created_at=CURRENT_TIMESTAMP WHERE id=?", (True, 'manual_accept', ann['id']))
                            # Reject overlapping
                            db.reject_overlapping_annotations(ann['sentence_id'], ann['start_char'], ann['end_char'], exclude_id=ann['id'])
                            
                            st.toast(f"Accepted: {ann['text_span']}")
                            # Move to next
                            if st.session_state.focus_index < len(pending_anns) - 1:
                                st.session_state.focus_index += 1
                            st.rerun()
                    
                if c2.button("❌ Reject", key=f"reject_focus_{ann['id']}", use_container_width=True):
                    # Soft Delete (Mark as Rejected)
                    db.execute_query("UPDATE annotations SET is_rejected=?, is_accepted=?, manual_edit_type=?, created_at=CURRENT_TIMESTAMP WHERE id=?", (True, False, 'manual_reject', ann['id']))
                    st.toast(f"Rejected: {ann['text_span']}")
                    st.rerun()
                    
                if c3.button("✏️ Edit", key=f"edit_focus_{ann['id']}", use_container_width=True):
                    st.session_state[f"editing_{ann['id']}"] = True

                # Edit Interface
                if st.session_state.get(f"editing_{ann['id']}", False):
                    render_edit_interface(ann, current_sentence['text'], all_labels, "focus_edit")

                # Navigation
                st.markdown("<br>", unsafe_allow_html=True)
                n1, n2, n3 = st.columns([1, 2, 1])
                if n1.button("⬅️ Previous Annotation"):
                    st.session_state.focus_index = max(0, st.session_state.focus_index - 1)
                    st.rerun()
                
                with n2:
                    st.progress((st.session_state.focus_index + 1) / len(pending_anns))
                    st.caption(f"Annotation {st.session_state.focus_index + 1} of {len(pending_anns)}")
                    
                if n3.button("Next Annotation ➡️"):
                    st.session_state.focus_index = min(len(pending_anns) - 1, st.session_state.focus_index + 1)
                    st.rerun()
                    
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            # --- LIST MODE (Default) ---
            for ann in pending_anns:
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    
                    with c1:
                        # Color map for entity labels
                        label_colors = {
                            'PERSON': '#007bff',  # blue
                            'ORG': '#6f42c1',     # purple
                            'GPE': '#28a745',     # green
                            'LOCATION': '#17a2b8',# teal
                            'FACILITY': '#fd7e14',# orange
                            'LEG_REFS': '#e83e8c',# pink
                            'PUBLIC_DOCS': '#343a40', # dark gray
                            'DATE': '#f6c23e',    # yellow
                        }
                        label = ann['label']
                        color = label_colors.get(label, '#444')
                        # Show index badge, then the text_span (main annotation text), then the label pill and the rest
                        st.markdown(f"<span class='entity-badge'>#{ann['display_index']}</span> <b>{ann['text_span']}</b> <span style='display:inline-block;background:{color};color:#fff;padding:1px 8px;border-radius:6px;font-weight:600;font-size:0.92em;margin-left:8px;margin-right:8px;'>{label}</span> <span style='color:#888;'>{ann['source_agent']} • {ann['confidence']:.2f}</span>", unsafe_allow_html=True)
                    
                    with c2:
                        confirm_key = f"confirm_accept_{ann['id']}"
                        if st.session_state.get(confirm_key, False):
                            st.warning("⚠️ Overlap!")
                            if st.button("Confirm", key=f"conf_btn_{ann['id']}"):
                                db.execute_query("UPDATE annotations SET is_accepted=?, manual_edit_type=?, created_at=CURRENT_TIMESTAMP WHERE id=?", (True, 'manual_accept', ann['id']))
                                # Reject overlapping (exclude self)
                                db.reject_overlapping_annotations(ann['sentence_id'], ann['start_char'], ann['end_char'], exclude_id=ann['id'])
                                
                                if ann.get('vector'):
                                    db.update_entity_centroid(ann['text_span'], ann['label'], ann['vector'])
                                del st.session_state[confirm_key]
                                st.rerun()
                            if st.button("Cancel", key=f"canc_btn_{ann['id']}"):
                                del st.session_state[confirm_key]
                                st.rerun()
                        else:
                            if st.button("✅", key=f"accept_{ann['id']}", help="Accept"):
                                overlaps = db.check_overlapping_annotations(ann['sentence_id'], ann['start_char'], ann['end_char'], exclude_id=ann['id'])
                                if overlaps:
                                    st.session_state[confirm_key] = True
                                    st.rerun()
                                else:
                                    db.execute_query("UPDATE annotations SET is_accepted=?, manual_edit_type=?, created_at=CURRENT_TIMESTAMP WHERE id=?", (True, 'manual_accept', ann['id']))
                                    # No overlaps to reject
                                    
                                    if ann.get('vector'):
                                        db.update_entity_centroid(ann['text_span'], ann['label'], ann['vector'])
                                    st.rerun()
                    
                    with c3:
                        if st.button("❌", key=f"reject_{ann['id']}", help="Reject"):
                            # Soft Delete
                            db.execute_query("UPDATE annotations SET is_rejected=?, is_accepted=?, manual_edit_type=?, created_at=CURRENT_TIMESTAMP WHERE id=?", (True, False, 'manual_reject', ann['id']))
                            st.rerun()
                            
                    with c4:
                        if st.button("✏️", key=f"edit_{ann['id']}", help="Edit"):
                            st.session_state[f"editing_{ann['id']}"] = True
                            
                    # Edit Mode Expander
                    if st.session_state.get(f"editing_{ann['id']}", False):
                        with st.expander("Edit Annotation", expanded=True):
                            render_edit_interface(ann, current_sentence['text'], all_labels, "list_edit")

    st.divider()

    # --- 3. Manual Annotation Section ---
    with st.expander("➕ Add Missing Annotation", expanded=False):
        st.caption("Use the Start/End indices shown on the words below to select the entity.")
        
        # 1. Render Tokenized Text (Cached)
        st.markdown(render_tokenized_text(current_sentence['text']), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Inputs
        c1, c2, c3 = st.columns([1, 1, 2])
        
        # Initialize state for manual add reset counter
        if "manual_reset_counter" not in st.session_state: 
            st.session_state.manual_reset_counter = 0
        
        with c1:
            m_start = st.number_input("Start Index", min_value=0, max_value=len(current_sentence['text']), key=f"manual_start_{st.session_state.manual_reset_counter}")
        with c2:
            m_end = st.number_input("End Index", min_value=0, max_value=len(current_sentence['text']), key=f"manual_end_{st.session_state.manual_reset_counter}")
        with c3:
            new_label = st.selectbox("Label", all_labels, key="new_ann_label")
            
        # 3. Preview & Logic
        if m_end > m_start:
            preview_text = current_sentence['text'][m_start:m_end]
            st.info(f"Preview: **{preview_text}**")
        else:
            st.warning("End Index must be greater than Start Index.")
            preview_text = ""

        # 4. Actions
        ac1, ac2 = st.columns([1, 1])
        if ac1.button("💾 Save New Annotation", type="primary", disabled=(not preview_text)):
            insert_cursor = db.execute_query("""
                INSERT INTO annotations (sentence_id, text_span, label, start_char, end_char, confidence, source_agent, is_accepted, manual_edit_type)
                VALUES (?, ?, ?, ?, ?, 1.0, 'human_manual', ?, 'manual_add')
                RETURNING id
            """, (current_sentence['id'], preview_text, new_label, m_start, m_end, True))
            # Πάρε το id του νέου annotation
            new_ann_id = None
            try:
                row = insert_cursor.fetchone()
                if row:
                    new_ann_id = row[0] if isinstance(row, tuple) else row['id']
            except Exception:
                try:
                    new_ann_id = insert_cursor.lastrowid
                except Exception:
                    pass
            db.reject_overlapping_annotations(current_sentence['id'], m_start, m_end, exclude_id=new_ann_id)
            st.toast("New annotation added successfully!")
            st.session_state.manual_reset_counter += 1
            st.rerun()
            
        if ac2.button("Cancel", key="cancel_manual_add"):
            st.rerun()

    st.divider()

    # --- 4. Ask LLM Assistant (Inside Fragment) ---
    if "llm_expanded" not in st.session_state:
        st.session_state["llm_expanded"] = False

    def keep_llm_open():
        st.session_state["llm_expanded"] = True

    # Clear LLM answer if sentence changes
    if "llm_last_sentence_id" not in st.session_state:
        st.session_state["llm_last_sentence_id"] = current_sentence['id']
    elif st.session_state["llm_last_sentence_id"] != current_sentence['id']:
        st.session_state["llm_last_sentence_id"] = current_sentence['id']
        if "llm_answer" in st.session_state:
            del st.session_state["llm_answer"]

    with st.expander("🤖 Ask LLM Assistant", expanded=st.session_state["llm_expanded"]):
        st.caption("Ask an AI expert about this sentence or a specific part of it.")
        
        # Model Selection
        available_models = {
            "Google: Gemini 2.5 Flash (Default)": "google/gemini-2.5-flash",
            "xAI: Grok Code Fast 1 ($0.20/M in, $1.50/M out)": "x-ai/grok-code-fast-1",
            "Anthropic: Claude Sonnet 4.5 ($3/M in, $15/M out)": "anthropic/claude-sonnet-4.5",
            "DeepSeek: DeepSeek V3.2 ($0.224/M in, $0.32/M out)": "deepseek/deepseek-v3.2",
            "xAI: Grok 4.1 Fast ($0.20/M in, $0.50/M out)": "x-ai/grok-4.1-fast",
            "Anthropic: Claude Opus 4.5 ($5/M in, $25/M out)": "anthropic/claude-opus-4.5",
            "DeepSeek: DeepSeek V3 0324 ($0.20/M in, $0.88/M out)": "deepseek/deepseek-chat-v3-0324",
            "Google: Gemini 3 Pro Preview ($2/M in, $12/M out)": "google/gemini-3-pro-preview",
            "Anthropic: Claude Haiku 4.5 ($1/M in, $5/M out)": "anthropic/claude-haiku-4.5",
            "OpenAI: GPT-5 Mini ($0.25/M in, $2/M out)": "openai/gpt-5-mini",
            "OpenAI: GPT-5.2 ($1.75/M in, $14/M out)": "openai/gpt-5.2",
            "NVIDIA: Nemotron 3 Nano 30B A3B (Free)": "nvidia/nemotron-3-nano-30b-a3b:free",
            "Z.AI: GLM 4.6 ($0.39/M in, $1.90/M out)": "z-ai/glm-4.6"
        }
        
        selected_model_name = st.selectbox("Select AI Model", list(available_models.keys()), index=0, key="llm_model_select")
        selected_model_id = available_models[selected_model_name]

        llm_mode = st.radio(
            "Context Scope", 
            ["Whole Sentence", "Specific Segment"], 
            horizontal=True,
            key="llm_mode",
            on_change=keep_llm_open
        )
        
        llm_segment = None
        if llm_mode == "Specific Segment":
            lc1, lc2 = st.columns(2)
            with lc1:
                l_start = st.number_input("Start Index (LLM)", min_value=0, max_value=len(current_sentence['text']), value=0, key="llm_start", on_change=keep_llm_open)
            with lc2:
                l_end = st.number_input("End Index (LLM)", min_value=0, max_value=len(current_sentence['text']), value=len(current_sentence['text']), key="llm_end", on_change=keep_llm_open)
                
            if l_end > l_start:
                llm_segment = current_sentence['text'][l_start:l_end]
                st.info(f"Selected Segment: **{llm_segment}**")
            else:
                st.warning("Invalid segment range.")
                
        with st.form(key="llm_form"):
            # Check for API Key (Secrets -> Env -> Input)
            api_key_input = None
            has_secret = False
            try:
                if st.secrets.get("OPENROUTER_API_KEY"):
                    has_secret = True
            except:
                pass
                
            if not has_secret and not os.getenv("OPENROUTER_API_KEY"):
                st.warning("⚠️ OPENROUTER_API_KEY not found in Secrets or Environment.")
                api_key_input = st.text_input("Enter OpenRouter API Key:", type="password", help="Get one from openrouter.ai")
            
            user_question = st.text_area("Your Question", placeholder="π.χ. ειναι το επιλεγμένο κείμενο ORG ή εινια κατι άλλο?")
            submit_button = st.form_submit_button("🚀 Send Prompt", type="primary")
            
        if submit_button:
            st.session_state["llm_expanded"] = True
            if not user_question:
                 st.warning("Please enter a question.")
            else:
                # Use input key if env var is missing
                key_to_use = api_key_input if api_key_input else None
                
                # Create a placeholder for the streaming response
                st.divider()
                st.success("### AI Response")
                
                judge = LLMJudge(api_key=key_to_use, model=selected_model_id)
                
                # Use st.write_stream for the "GenAI" feel
                full_response = st.write_stream(judge.stream_question(current_sentence['text'], user_question, llm_segment))
                st.session_state['llm_answer'] = full_response

        if "llm_answer" in st.session_state and not submit_button:
            st.divider()
            if "Error" in st.session_state['llm_answer'] or "Exception" in st.session_state['llm_answer']:
                st.error(st.session_state['llm_answer'])
            else:
                st.success("### AI Response")
                st.markdown(st.session_state['llm_answer'])

    st.divider()

    # --- 5. Accepted Annotations (Compact View) ---
    if accepted_anns:
        st.subheader("✅ Accepted Entities")
        cols = st.columns(4)
        for i, ann in enumerate(accepted_anns):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{ann['text_span']}**")
                    
                    # Display Label and Edit Type
                    edit_type = ann.get('manual_edit_type')
                    caption_text = f"{ann['label']}"
                    if edit_type:
                        caption_text += f"\n\n*{edit_type}*"
                    st.caption(caption_text)
                    
                    b1, b2, b3 = st.columns(3)
                    if b1.button("↩️", key=f"undo_{ann['id']}", help="Return to Pending"):
                         db.execute_query("UPDATE annotations SET is_accepted=?, manual_edit_type=?, created_at=CURRENT_TIMESTAMP WHERE id=?", (False, None, ann['id']))
                         st.rerun()
                    if b2.button("🧠", key=f"mem_acc_{ann['id']}", help="Propagate (Memory)"):
                         st.session_state['memory_search_term'] = ann['text_span']
                         st.session_state['memory_search_label'] = ann['label']
                         st.session_state['memory_search_id'] = ann['id']
                         st.session_state['memory_source_status'] = 'accepted'
                         st.rerun()
                    if b3.button("🔍", key=f"fuzzy_acc_{ann['id']}", help="Fuzzy Search in Raw Text"):
                         st.session_state['fuzzy_search_term'] = ann['text_span']
                         st.session_state['fuzzy_search_label'] = ann['label']
                         st.rerun()

    # --- 6. Rejected Annotations (Compact View) ---
    if rejected_anns:
        st.subheader("🗑️ Rejected Entities")
        cols = st.columns(4)
        for i, ann in enumerate(rejected_anns):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"~~{ann['text_span']}~~")
                    
                    # Display Label and Edit Type
                    edit_type = ann.get('manual_edit_type')
                    caption_text = f"{ann['label']}"
                    if edit_type:
                        caption_text += f"\n\n*{edit_type}*"
                    st.caption(caption_text)
                    
                    if st.button("↩️ Restore", key=f"restore_{ann['id']}", help="Return to Pending"):
                        # Overlap check πριν το restore
                        existing_anns = db.get_annotations_for_sentence(ann['sentence_id'])
                        non_rejected = [a for a in existing_anns if not a.get('is_rejected', 0) and a['id'] != ann['id']]
                        overlap = False
                        for a in non_rejected:
                            if (ann['start_char'] < a['end_char']) and (ann['end_char'] > a['start_char']):
                                overlap = True
                                break
                        if overlap:
                            st.error("Δεν μπορεί να γίνει restore: Υπάρχει overlap με ενεργή οντότητα.")
                        else:
                            db.execute_query("UPDATE annotations SET is_accepted=?, is_rejected=?, manual_edit_type=? WHERE id=?", (False, False, None, ann['id']))
                            st.rerun()
                    
                    if st.button("🧠", key=f"mem_rej_{ann['id']}", help="Propagate (Memory)"):
                        st.session_state['memory_search_term'] = ann['text_span']
                        st.session_state['memory_search_label'] = ann['label']
                        st.session_state['memory_search_id'] = ann['id']
                        st.session_state['memory_source_status'] = 'rejected'
                        st.rerun()

# Call the fragment
render_main_annotation_interface(current_sentence, all_labels)

# --- 7. Memory Propagation Interface ---
@st.fragment
def render_memory_propagation_interface():
    # Detect DB Type for query syntax
    is_postgres = "psycopg2" in str(type(db.conn))
    ph = "%s" if is_postgres else "?"
    true_val = "TRUE" if is_postgres else "1"
    false_val = "FALSE" if is_postgres else "0"

    # (Monkey patch removed: Using optimized db_manager implementation directly)

    if 'memory_search_term' not in st.session_state:
        return

    st.divider()
    st.subheader(f"🧠 Memory Propagation: '{st.session_state['memory_search_term']}'")
    
    # Close Button
    if st.button("Close Memory View"):
        del st.session_state['memory_search_term']
        del st.session_state['memory_search_label']
        del st.session_state['memory_search_id']
        st.rerun()
        
    # Search Logic
    term = st.session_state['memory_search_term']
    label = st.session_state['memory_search_label']
    source_id = st.session_state['memory_search_id']
    source_status = st.session_state.get('memory_source_status', 'accepted') # Default to accepted if missing

    # Controls Row
    mc1, mc2, mc3 = st.columns([2, 2, 2])
    
    with mc1:
        # Threshold Slider
        threshold = st.slider("Cosine Similarity Threshold", 0.5, 1.0, 0.9, 0.01)
    
    with mc2:
        # Search Mode Selection
        search_mode = st.radio("Search Mode", ["Adaptive Centroid", "Original Vector"], index=0, horizontal=True, help="Adaptive: Learns from your accepts. Original: Strict match to initial selection.")

    with mc3:
        # Label Filter
        label_filter_mode = st.radio("Label Filter", ["Same", "Diff", "All"], index=0, horizontal=True, help=f"Filter candidates by label.")
        
        # Force Label Update Checkbox (Only visible if Diff or All is selected AND Status is Accepted)
        force_label_update = False
        if label_filter_mode in ["Diff", "All"] and source_status == 'accepted':
            force_label_update = st.checkbox(f"Accept as '{label}'", value=False, help=f"If checked, accepted entities will be converted to '{label}' regardless of their original label.")

    # Get Source Vector
    # We need to fetch the vector of the accepted annotation first
    source_vector = None
    
    # 1. Try Memory Centroid (Adaptive Search)
    if search_mode == "Adaptive Centroid":
        cursor = db.execute_query(f"SELECT vector, count FROM memory_centroids WHERE source_annotation_id={ph}", (source_id,))
        row = cursor.fetchone()
        
        if row and row['vector']:
            source_vector = row['vector']
            count = row['count']
            st.caption(f"🧠 Using **Adaptive Memory Centroid** (Based on {count} examples)")
        else:
            st.caption("ℹ️ No centroid yet. Falling back to Original Vector.")
            
    # 2. Fallback or Explicit Original Vector
    if source_vector is None:
        cursor = db.execute_query(f"SELECT vector FROM annotations WHERE id={ph}", (source_id,))
        row = cursor.fetchone()
        source_vector = row['vector'] if row else None
        if search_mode == "Original Vector":
            st.caption("🔹 Using **Original Annotation Vector** (Static)")
    
    if source_vector is None:
        st.error("This annotation has no vector embedding stored. Cannot perform vector search.")
    else:
        # --- Caching Logic for Memory Search ---
        # We cache the raw candidates to avoid re-running the vector search on every interaction (checkbox click)
        mem_cache_key = (source_id, threshold, search_mode, label_filter_mode)
        
        if 'memory_candidates_cache' not in st.session_state or st.session_state.get('memory_cache_params') != mem_cache_key:
            with st.spinner("🧠 Searching Memory..."):
                candidates = db.get_similar_pending_annotations(
                    source_vector, 
                    threshold=threshold, 
                    limit=None,
                    label_filter=label,
                    filter_mode=label_filter_mode
                )
            st.session_state['memory_candidates_cache'] = candidates
            st.session_state['memory_cache_params'] = mem_cache_key
        else:
            candidates = st.session_state['memory_candidates_cache']
        
        # Filter out hidden (declined) candidates
        if 'memory_hidden_ids' not in st.session_state:
            st.session_state['memory_hidden_ids'] = set()
            
        visible_candidates = [c for c in candidates if c['id'] not in st.session_state['memory_hidden_ids']]
        
        # (Label filtering is now done at SQL level in _get_similar_pending_annotations)
        
        if not visible_candidates:
            st.info(f"No pending annotations found with similarity >= {threshold}.")
        else:
            st.write(f"Found **{len(visible_candidates)}** similar pending annotations.")
            
            # --- Pagination & Controls ---
            c_page, c_limit = st.columns([3, 1])
            with c_limit:
                ITEMS_PER_PAGE = st.selectbox("Items per page", [5, 10, 20, 50, 100, 200, 500], index=0, key="mem_per_page_select")
            
            if 'memory_page' not in st.session_state: st.session_state.memory_page = 0
            
            total_pages = (len(visible_candidates) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
            # Ensure current page is valid
            if st.session_state.memory_page >= total_pages: st.session_state.memory_page = 0
            
            start_idx = st.session_state.memory_page * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            page_candidates = visible_candidates[start_idx:end_idx]
            
            with c_page:
                st.caption(f"Page {st.session_state.memory_page + 1} of {total_pages}")
            
            # --- Bulk Selection ---
            bs1, bs2, _ = st.columns([1, 1, 4])
            if bs1.button("Select All", key=f"mem_sel_all_{st.session_state.memory_page}"):
                for c in page_candidates:
                    st.session_state[f"mem_chk_{c['id']}"] = True
                st.rerun()
            
            if bs2.button("Deselect All", key=f"mem_desel_all_{st.session_state.memory_page}"):
                for c in page_candidates:
                    st.session_state[f"mem_chk_{c['id']}"] = False
                st.rerun()
            
            selected_mem_indices = []

            for i, cand in enumerate(page_candidates):
                with st.container(border=True):
                    col_check, col_content = st.columns([0.5, 11])
                    
                    with col_check:
                        chk_key = f"mem_chk_{cand['id']}"
                        if chk_key not in st.session_state:
                            st.session_state[chk_key] = False
                            
                        is_selected = st.checkbox("Select", key=chk_key, label_visibility="collapsed")
                        if is_selected:
                            selected_mem_indices.append(i)

                    with col_content:
                        # Initialize session state for boundaries if not present
                        if f"mem_start_{cand['id']}" not in st.session_state:
                            st.session_state[f"mem_start_{cand['id']}"] = cand.get('start_char', 0)
                        if f"mem_end_{cand['id']}" not in st.session_state:
                            st.session_state[f"mem_end_{cand['id']}"] = cand.get('end_char', 0)
                        
                        # Always read from session state to reflect updates from callbacks
                        c_start = st.session_state[f"mem_start_{cand['id']}"]
                        c_end = st.session_state[f"mem_end_{cand['id']}"]
                        
                        c_text = cand['sentence_text']
                        # --- LEG-REFS: Ειδικό κουμπί για επέκταση σε τίτλο ---
                        label_val = str(cand.get('label', '')).strip().lower()
                        if label_val == 'leg-refs':
                            st.button("Επέκταση σε τίτλο (auto)", 
                                      key=f"expand_title_mem_{cand['id']}",
                                      on_click=expand_title_callback,
                                      args=(f"mem_start_{cand['id']}", f"mem_end_{cand['id']}", c_text, 'leg-refs'))
                        elif label_val == 'public-docs':
                            st.button("Επέκταση σε τίτλο (auto)", 
                                      key=f"expand_title_mem_pubdocs_{cand['id']}",
                                      on_click=expand_title_callback,
                                      args=(f"mem_start_{cand['id']}", f"mem_end_{cand['id']}", c_text, 'public-docs'))
                        else:
                            st.caption(f"[DEBUG] Label: '{cand.get('label')}' (κουμπί δεν εμφανίζεται)")
                        
                        # Highlight logic
                        if c_start is not None and c_end is not None and c_start >=  0 and c_end <= len(c_text):
                            # Precise highlighting using indices
                            prefix = html_lib.escape(c_text[:c_start])
                            target = html_lib.escape(c_text[c_start:c_end])
                            suffix = html_lib.escape(c_text[c_end:])
                            
                            display_text = (
                                f"{prefix}<span style='background-color:#d1c4e9; font-weight:bold; border:1px solid #9575cd;'>{target}</span>{suffix}"
                            )
                        else:
                            # Fallback to regex if indices are missing
                            c_span = cand['text_span']
                            pattern = re.compile(re.escape(c_span), re.IGNORECASE)
                            display_text = pattern.sub(f"<span style='background-color:#d1c4e9; font-weight:bold; border:1px solid #9575cd;'>{c_span}</span>", c_text)
                        
                        st.markdown(display_text, unsafe_allow_html=True)
                        st.caption(f"Similarity: **{cand['similarity']:.4f}** | Label: `{cand['label']}` | Sentence ID: `{cand['sentence_id']}`")
                        
                        # Boundary Controls for Memory
                        mb1, mb2, mb3, mb4 = st.columns(4)
                        with mb1:
                            st.button("⬅️ Expand", key=f"mem_el_{cand['id']}",
                                      on_click=update_boundary_callback,
                                      args=(cand['id'], "mem", c_text, "expand_left"))
                        with mb2:
                            st.button("Shrink ➡️", key=f"mem_sl_{cand['id']}",
                                      on_click=update_boundary_callback,
                                      args=(cand['id'], "mem", c_text, "shrink_left"))
                        with mb3:
                            st.button("⬅️ Shrink", key=f"mem_sr_{cand['id']}",
                                      on_click=update_boundary_callback,
                                      args=(cand['id'], "mem", c_text, "shrink_right"))
                        with mb4:
                            st.button("Expand ➡️", key=f"mem_er_{cand['id']}",
                                      on_click=update_boundary_callback,
                                      args=(cand['id'], "mem", c_text, "expand_right"))
                    
            # --- Bulk Actions ---
            st.divider()
            
            # Dynamic Button Layout based on Source Status
            if source_status == 'rejected':
                 # If source is rejected, emphasize REJECT action
                 # We hide "Accept Selected" as per user request to avoid confusion ("we reject not accept in rejected memory")
                 ma1, ma2 = st.columns([1, 4])
                 
                 # Reject Bulk (Primary)
                 if ma1.button(f"🚫 Reject Selected ({len(selected_mem_indices)})", key="mem_reject_bulk", type="primary", disabled=len(selected_mem_indices)==0):
                    with st.spinner("Processing Rejection..."):
                        cursor = db.conn.cursor()
                        count_rejected = 0
                        try:
                            for idx in selected_mem_indices:
                                cand = page_candidates[idx]
                                c_start = st.session_state.get(f"mem_start_{cand['id']}", cand.get('start_char', 0))
                                c_end = st.session_state.get(f"mem_end_{cand['id']}", cand.get('end_char', 0))
                                c_text = cand['sentence_text']
                                final_span = c_text[c_start:c_end]
                                old_label = cand.get('label')
                                # No label update on reject
                                manual_edit_type = "memory_reject"
                                cursor.execute(
                                    f"UPDATE annotations SET is_rejected={true_val}, is_accepted={false_val}, label={ph}, start_char={ph}, end_char={ph}, text_span={ph}, manual_edit_type={ph}, created_at=CURRENT_TIMESTAMP WHERE id={ph}",
                                    (old_label, c_start, c_end, final_span, manual_edit_type, cand['id'])
                                )
                                # Cleanup Session State
                                if f"mem_start_{cand['id']}" in st.session_state: del st.session_state[f"mem_start_{cand['id']}"]
                                if f"mem_end_{cand['id']}" in st.session_state: del st.session_state[f"mem_end_{cand['id']}"]
                                count_rejected += 1
                            db.conn.commit()
                        except Exception as e:
                            db.conn.rollback()
                            st.error(f"Error during bulk reject: {e}")
                            
                        st.toast(f"🚫 Rejected {count_rejected} annotations!")
                        # Invalidate cache to force refresh
                        if 'memory_candidates_cache' in st.session_state:
                            del st.session_state['memory_candidates_cache']
                        st.rerun()

            else:
                # Default Layout (Source is Accepted)
                ma1, ma2, ma3 = st.columns(3)
            
                # Accept Bulk (existing)
                if ma1.button(f"✅ Accept Selected ({len(selected_mem_indices)})", key="mem_accept_bulk", type="primary", disabled=len(selected_mem_indices)==0):
                    with st.spinner("Processing Memory Acceptance..."):
                        # --- OPTIMIZATION: Single Transaction for Bulk Update ---
                        # We bypass db.execute_query to avoid committing after every single row.
                        cursor = db.conn.cursor()
                        count_accepted = 0
                        
                        try:
                            for idx in selected_mem_indices:
                                cand = page_candidates[idx]
                                c_start = st.session_state.get(f"mem_start_{cand['id']}", cand.get('start_char', 0))
                                c_end = st.session_state.get(f"mem_end_{cand['id']}", cand.get('end_char', 0))
                                c_text = cand['sentence_text']
                                final_span = c_text[c_start:c_end]
                                
                                # 1. Update Annotation (Accept)
                                old_label = cand.get('label')
                                target_label = label if (force_label_update or old_label == label) else old_label
                                
                                if target_label != old_label:
                                    manual_edit_type = f"memory_accept_converted_{old_label}"
                                else:
                                    manual_edit_type = "memory_accept"
                                    
                                cursor.execute(
                                    f"UPDATE annotations SET is_accepted={true_val}, label={ph}, start_char={ph}, end_char={ph}, text_span={ph}, manual_edit_type={ph}, created_at=CURRENT_TIMESTAMP WHERE id={ph}", 
                                    (target_label, c_start, c_end, final_span, manual_edit_type, cand['id'])
                                )
                                
                                # 2. Reject Overlapping (Inline Logic)
                                cursor.execute(f"""
                                    UPDATE annotations 
                                    SET is_rejected = {true_val}, is_accepted = {false_val}
                                    WHERE sentence_id = {ph} 
                                    AND (is_rejected = {false_val} OR is_rejected IS NULL)
                                    AND start_char < {ph} 
                                    AND end_char > {ph}
                                    AND id != {ph}
                                """, (cand['sentence_id'], c_end, c_start, cand['id']))
                                
                                # Note: We skip per-item Centroid Update here for performance (saves 2 queries per item).
                                # The centroid will update on the next single-item accept or we can implement batch update later.

                                # Cleanup Session State
                                if f"mem_start_{cand['id']}" in st.session_state: del st.session_state[f"mem_start_{cand['id']}"]
                                if f"mem_end_{cand['id']}" in st.session_state: del st.session_state[f"mem_end_{cand['id']}"]
                                count_accepted += 1
                            
                            # --- 3. Bulk Centroid Update (Optimized) ---
                            source_id = st.session_state.get('memory_search_id')
                            if source_id and count_accepted > 0:
                                try:
                                    # Get IDs of accepted items
                                    accepted_ids = [page_candidates[i]['id'] for i in selected_mem_indices]
                                    placeholders = ",".join([ph] * len(accepted_ids))
                                    
                                    # Fetch vectors of accepted items
                                    cursor.execute(f"SELECT vector FROM annotations WHERE id IN ({placeholders})", tuple(accepted_ids))
                                    vec_rows = cursor.fetchall()
                                    
                                    new_vectors = []
                                    for r in vec_rows:
                                        if r['vector']:
                                            # Handle memoryview/bytes
                                            v_blob = r['vector']
                                            if isinstance(v_blob, memoryview): v_blob = bytes(v_blob)
                                            new_vectors.append(np.frombuffer(v_blob, dtype=np.float32))
                                    
                                    if new_vectors:
                                        batch_sum = np.sum(new_vectors, axis=0)
                                        batch_count = len(new_vectors)
                                        
                                        # Fetch existing centroid
                                        cursor.execute(f"SELECT vector, count FROM memory_centroids WHERE source_annotation_id = {ph}", (source_id,))
                                        c_row = cursor.fetchone()
                                        
                                        if c_row:
                                            # Update existing
                                            old_blob = c_row['vector']
                                            if isinstance(old_blob, memoryview): old_blob = bytes(old_blob)
                                            old_vec = np.frombuffer(old_blob, dtype=np.float32)
                                            old_count = c_row['count']
                                            
                                            new_centroid = (old_vec * old_count + batch_sum) / (old_count + batch_count)
                                            new_count = old_count + batch_count
                                            
                                            cursor.execute(f"""
                                                UPDATE memory_centroids 
                                                SET vector = {ph}, count = {ph}, last_updated = CURRENT_TIMESTAMP 
                                                WHERE source_annotation_id = {ph}
                                            """, (new_centroid.tobytes(), new_count, source_id))
                                        else:
                                            # Insert new
                                            new_centroid = batch_sum / batch_count
                                            cursor.execute(f"""
                                                INSERT INTO memory_centroids (source_annotation_id, text_span, label, vector, count)
                                                VALUES ({ph}, {ph}, {ph}, {ph}, {ph})
                                            """, (source_id, term, label, new_centroid.tobytes(), batch_count))
                                            
                                except Exception as e:
                                    print(f"⚠️ Bulk Centroid Update Failed: {e}")

                            # Commit all changes at once
                            db.conn.commit()
                            
                        except Exception as e:
                            db.conn.rollback()
                            st.error(f"Error during bulk accept: {e}")
                            print(f"Bulk Accept Error: {e}")
                        
                        # Invalidate Cache to force refresh from DB
                        if 'memory_candidates_cache' in st.session_state:
                            del st.session_state['memory_candidates_cache']

                        st.toast(f"✅ Accepted {count_accepted} annotations!")
                        st.rerun()

                # Reject Bulk (new)
                if ma3.button(f"🚫 Reject Selected ({len(selected_mem_indices)})", key="mem_reject_bulk", type="primary", disabled=len(selected_mem_indices)==0):
                    with st.spinner("Processing Rejection..."):
                        cursor = db.conn.cursor()
                        count_rejected = 0
                        try:
                            for idx in selected_mem_indices:
                                cand = page_candidates[idx]
                                s_key = f"mem_start_{cand['id']}"
                                e_key = f"mem_end_{cand['id']}"
                                c_start = st.session_state.get(s_key, cand.get('start_char', 0))
                                c_end = st.session_state.get(e_key, cand.get('end_char', 0))
                                c_text = cand['sentence_text']
                                final_span = c_text[c_start:c_end]
                                old_label = cand.get('label')
                                target_label = label if (force_label_update or old_label == label) else old_label
                                manual_edit_type = "memory_reject"
                                cursor.execute(
                                    f"UPDATE annotations SET is_rejected={true_val}, is_accepted={false_val}, label={ph}, start_char={ph}, end_char={ph}, text_span={ph}, manual_edit_type={ph}, created_at=CURRENT_TIMESTAMP WHERE id={ph}",
                                    (target_label, c_start, c_end, final_span, manual_edit_type, cand['id'])
                                )
                                # Cleanup Session State
                                if s_key in st.session_state: del st.session_state[s_key]
                                if e_key in st.session_state: del st.session_state[e_key]
                                count_rejected += 1
                            db.conn.commit()
                        except Exception as e:
                            db.conn.rollback()
                            st.error(f"Error during bulk reject: {e}")
                        st.toast(f"🚫 Rejected {count_rejected} annotations!")
                        # Invalidate cache to force refresh
                        if 'memory_candidates_cache' in st.session_state:
                            del st.session_state['memory_candidates_cache']
                        st.rerun()

                if total_pages > 1:
                    pc1, pc2, pc3 = st.columns([1, 2, 1])
                    if pc1.button("⬅️ Prev", key="mem_prev", disabled=(st.session_state.memory_page == 0)):
                        st.session_state.memory_page -= 1
                        st.rerun()
                    pc2.markdown(f"<div style='text-align:center'>Page {st.session_state.memory_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
                    if pc3.button("Next ➡️", key="mem_next", disabled=(st.session_state.memory_page >= total_pages - 1)):
                        st.session_state.memory_page += 1
                        st.rerun()

render_memory_propagation_interface()

# --- 7. Fuzzy Search Interface ---
@st.fragment
def render_fuzzy_search_interface():
    if 'fuzzy_search_term' not in st.session_state:
        return
    st.divider()
    st.subheader(f"🔍 Fuzzy Search: '{st.session_state['fuzzy_search_term']}'")
    
    if st.button("Close Fuzzy Search"):
        del st.session_state['fuzzy_search_term']
        del st.session_state['fuzzy_search_label']
        st.rerun()
        
    if fuzz is None:
        st.error("RapidFuzz library is not installed. Please install it using `pip install rapidfuzz`.")
    else:
        term = st.session_state['fuzzy_search_term']
        label = st.session_state['fuzzy_search_label']
        
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            threshold = st.slider("Fuzzy Threshold", 30, 100, 85, 1)
        with fc2:
            len_threshold = st.slider("Min Length %", 10, 100, 80, 5, help="Match must be at least this % of the search term's length")
        with fc3:
            # Advanced Filters for Overlap
            st.caption("Allow search in:")
            include_rejected = st.checkbox("Rejected Areas", value=True, help="Search in text marked as Rejected.")
            include_pending = st.checkbox("Pending Areas", value=False, help="Search in text marked as Pending.")
            include_accepted = st.checkbox("Accepted Areas", value=False, help="Search in text marked as Accepted.")
        
        # OPTIMIZATION: Instead of fetching ALL sentences, we fetch candidates using SQL LIKE
        @st.cache_data(ttl=600)
        def get_fuzzy_candidates(search_term):
            cursor = db.conn.cursor()
            words = [w for w in search_term.split() if len(w) > 2]
            if not words:
                cursor.execute("SELECT id, text FROM sentences LIMIT 1000")
                return cursor.fetchall()
            
            is_postgres = "psycopg2" in str(type(db.conn))
            ph = "%s" if is_postgres else "?"
            clauses = []
            params = []
            for w in words:
                if is_postgres:
                    clauses.append("text ILIKE " + ph)
                else:
                    clauses.append("text LIKE " + ph)
                params.append(f"%{w}%")
            
            query = f"SELECT id, text FROM sentences WHERE {' OR '.join(clauses)} LIMIT 10000"
            cursor.execute(query, tuple(params))
            return cursor.fetchall()

        all_sentences = get_fuzzy_candidates(term)
        cursor = db.conn.cursor()
        
        if not all_sentences:
            st.error("⚠️ No sentences found in the database!")
        else:
            # Fetch existing annotations to filter duplicates
            try:
                cursor.execute("SELECT sentence_id, start_char, end_char, is_accepted, is_rejected FROM annotations")
            except Exception:
                if hasattr(db, 'conn'): db.conn.rollback()
                cursor.execute("SELECT sentence_id, start_char, end_char, is_accepted, is_rejected FROM annotations")
                
            existing_anns = cursor.fetchall()
            existing_map = {}
            
            def get_status(r):
                if r['is_rejected']: return 'rejected'
                if r['is_accepted']: return 'accepted'
                return 'pending'

            for r in existing_anns:
                status = get_status(r)
                should_block = True
                
                # Rule: "include_X = True" means we WANT to find these areas, so they should NOT block.
                # If include_rejected is ON, then rejected annotations do NOT block (should_block = False).
                # If include_rejected is OFF, then rejected annotations DO block (should_block = True).
                
                if status == 'rejected': should_block = not include_rejected
                elif status == 'pending': should_block = not include_pending
                elif status == 'accepted': should_block = not include_accepted
                
                if should_block:
                    sid = r['sentence_id']
                    if sid not in existing_map: existing_map[sid] = []
                    existing_map[sid].append((r['start_char'], r['end_char']))

            # --- Caching Logic for Fuzzy Search ---
            fuz_cache_key = (term, threshold, len_threshold, include_rejected, include_pending, include_accepted)
            
            if 'fuzzy_candidates_cache' not in st.session_state or st.session_state.get('fuzzy_cache_params') != fuz_cache_key:
                fuzzy_candidates = []
                progress_bar = st.progress(0)
                total_sents = len(all_sentences)
                best_debug_score = 0
                best_debug_match = None
                rejected_reasons = {"length": 0, "secondary_score": 0, "duplicate": 0}
                
                def check_sentence(sent):
                    s_text = sent['text']
                    if not s_text or len(s_text) < len(term) * 0.5: return None
                    try:
                        alignment = fuzz.partial_ratio_alignment(term, s_text, score_cutoff=30)
                        if alignment: return (sent, alignment)
                    except: pass
                    return None

                with st.spinner("🔍 Searching Text (Parallel)..."):
                    # Use ThreadPoolExecutor for parallel fuzzy matching
                    with ThreadPoolExecutor() as executor:
                        raw_results = list(executor.map(check_sentence, all_sentences))
                    
                    for i, res in enumerate(raw_results):
                        if not res: continue
                        sent, alignment = res
                        s_text = sent['text']

                        if alignment.score > best_debug_score:
                            best_debug_score = alignment.score
                            best_debug_match = f"ID {sent['id']}: ...{s_text[max(0, alignment.dest_start-10):min(len(s_text), alignment.dest_end+10)]}..."

                        if alignment.score >= threshold:
                            d_start, d_end = alignment.dest_start, alignment.dest_end
                            while d_start > 0 and not s_text[d_start-1].isspace() and s_text[d_start-1] not in [',', '.', ';', ':', '!', '?']: d_start -= 1
                            while d_end < len(s_text) and not s_text[d_end].isspace() and s_text[d_end] not in [',', '.', ';', ':', '!', '?']: d_end += 1
                            
                            final_match = s_text[d_start:d_end]
                            if (d_end - d_start) < len(term) * (len_threshold / 100.0):
                                rejected_reasons["length"] += 1
                                continue
                            
                            if fuzz.ratio(term, final_match) < (threshold * 0.7):
                                rejected_reasons["secondary_score"] += 1
                                continue

                            is_duplicate = False
                            if sent['id'] in existing_map:
                                for ex_start, ex_end in existing_map[sent['id']]:
                                    if not (d_end <= ex_start or d_start >= ex_end):
                                        is_duplicate = True
                                        break
                            if is_duplicate:
                                rejected_reasons["duplicate"] += 1
                                continue

                            fuzzy_candidates.append({
                                'sentence_id': sent['id'],
                                'sentence_text': s_text,
                                'text_span': final_match,
                                'start_char': d_start,
                                'end_char': d_end,
                                'score': alignment.score
                            })
                        if i % 1000 == 0: progress_bar.progress(min((i + 1) / total_sents, 1.0))
                
                progress_bar.empty()
                fuzzy_candidates.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['fuzzy_candidates_cache'] = fuzzy_candidates
                st.session_state['fuzzy_cache_params'] = fuz_cache_key
                st.session_state['fuzzy_debug_info'] = (best_debug_score, best_debug_match, rejected_reasons)
            else:
                fuzzy_candidates = st.session_state['fuzzy_candidates_cache']
                best_debug_score, best_debug_match, rejected_reasons = st.session_state['fuzzy_debug_info']

            if not fuzzy_candidates:
                st.warning(f"No matches found above threshold {threshold}.")
                if best_debug_score > 0:
                    st.info(f"Best match score: {best_debug_score:.1f}\n\nSample: {best_debug_match}")
            else:
                st.success(f"Found **{len(fuzzy_candidates)}** potential matches!")
                
                # Pagination
                ITEMS_PER_PAGE = st.selectbox("Items per page", [5, 10, 20, 50, 100], index=1, key="fuz_limit")
                if 'fuzzy_page' not in st.session_state: st.session_state.fuzzy_page = 0
                total_pages = (len(fuzzy_candidates) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
                if st.session_state.fuzzy_page >= total_pages: st.session_state.fuzzy_page = 0
                
                page_candidates = fuzzy_candidates[st.session_state.fuzzy_page * ITEMS_PER_PAGE : (st.session_state.fuzzy_page + 1) * ITEMS_PER_PAGE]
                
                # Bulk Selection
                bs1, bs2, _ = st.columns([1, 1, 4])
                if bs1.button("Select All", key=f"fuz_all_{st.session_state.fuzzy_page}"):
                    for c in page_candidates: st.session_state[f"fuz_chk_{c['sentence_id']}_{c['start_char']}"] = True
                    st.rerun()
                if bs2.button("Deselect All", key=f"fuz_none_{st.session_state.fuzzy_page}"):
                    for c in page_candidates: st.session_state[f"fuz_chk_{c['sentence_id']}_{c['start_char']}"] = False
                    st.rerun()

                selected_indices = []
                for i, cand in enumerate(page_candidates):
                    with st.container(border=True):
                        col_check, col_content = st.columns([0.5, 11])
                        chk_key = f"fuz_chk_{cand['sentence_id']}_{cand['start_char']}"
                        if chk_key not in st.session_state: st.session_state[chk_key] = False
                        if col_check.checkbox("Select", key=chk_key, label_visibility="collapsed"):
                            selected_indices.append(i)

                        with col_content:
                            s_key, e_key = f"fuz_start_{cand['sentence_id']}_{cand['start_char']}", f"fuz_end_{cand['sentence_id']}_{cand['start_char']}"
                            if s_key not in st.session_state: st.session_state[s_key] = cand['start_char']
                            if e_key not in st.session_state: st.session_state[e_key] = cand['end_char']
                            
                            # Always read from session state to reflect updates from callbacks
                            c_start = st.session_state[s_key]
                            c_end = st.session_state[e_key]
                            
                            c_text = cand['sentence_text']
                            # --- LEG-REFS: Ειδικό κουμπί για επέκταση σε τίτλο ---
                            label_val = str(label).strip().lower()
                            if label_val == 'leg-refs':
                                st.button("Επέκταση σε τίτλο (auto)", 
                                          key=f"expand_title_fuz_{cand['sentence_id']}_{cand['start_char']}",
                                          on_click=expand_title_callback,
                                          args=(s_key, e_key, c_text, 'leg-refs'))
                            elif label_val == 'public-docs':
                                st.button("Επέκταση σε τίτλο (auto)", 
                                          key=f"expand_title_fuz_pubdocs_{cand['sentence_id']}_{cand['start_char']}",
                                          on_click=expand_title_callback,
                                          args=(s_key, e_key, c_text, 'public-docs'))
                            else:
                                st.caption(f"[DEBUG] Label: '{label}' (κουμπί δεν εμφανίζεται)")
                            prefix, target, suffix = html_lib.escape(c_text[:c_start]), html_lib.escape(c_text[c_start:c_end]), html_lib.escape(c_text[c_end:])
                            st.markdown(f"{prefix}<span style='background-color:#fff59d; font-weight:bold;'>{target}</span>{suffix}", unsafe_allow_html=True)
                            st.caption(f"Score: **{cand['score']:.1f}** | ID: `{cand['sentence_id']}`")
                            b1, b2, b3, b4 = st.columns(4)
                            b1.button("⬅️ Expand", key=f"el_{cand['sentence_id']}_{cand['start_char']}", on_click=update_boundary_callback, args=(f"{cand['sentence_id']}_{cand['start_char']}", "fuz", c_text, "expand_left"))
                            b2.button("Shrink ➡️", key=f"sl_{cand['sentence_id']}_{cand['start_char']}", on_click=update_boundary_callback, args=(f"{cand['sentence_id']}_{cand['start_char']}", "fuz", c_text, "shrink_left"))
                            b3.button("⬅️ Shrink", key=f"sr_{cand['sentence_id']}_{cand['start_char']}", on_click=update_boundary_callback, args=(f"{cand['sentence_id']}_{cand['start_char']}", "fuz", c_text, "shrink_right"))
                            b4.button("Expand ➡️", key=f"er_{cand['sentence_id']}_{cand['start_char']}", on_click=update_boundary_callback, args=(f"{cand['sentence_id']}_{cand['start_char']}", "fuz", c_text, "expand_right"))

                # Bulk Actions
                st.divider()
                ba1, ba2 = st.columns(2)
                should_execute = False
                
                if st.session_state.get("fuz_confirm_mode", False):
                    with ba1:
                        st.warning(f"⚠️ {st.session_state['fuz_overlap_count']} overlaps detected!")
                        if st.button("Confirm", key="fuz_conf_btn"):
                            should_execute = True
                            st.session_state["fuz_confirm_mode"] = False
                        if st.button("Cancel", key="fuz_canc_btn"):
                            st.session_state["fuz_confirm_mode"] = False
                            st.rerun()
                else:
                    if ba1.button(f"✅ Accept Selected ({len(selected_indices)})", key="fuz_accept_bulk", type="primary", disabled=len(selected_indices)==0):
                        overlap_count = 0
                        for idx in selected_indices:
                            cand = page_candidates[idx]
                            s_key = f"fuz_start_{cand['sentence_id']}_{cand['start_char']}"
                            e_key = f"fuz_end_{cand['sentence_id']}_{cand['start_char']}"
                            c_s = st.session_state.get(s_key, cand['start_char'])
                            c_e = st.session_state.get(e_key, cand['end_char'])
                            if db.check_overlapping_annotations(cand['sentence_id'], c_s, c_e):
                                overlap_count += 1
                                
                        if overlap_count > 0:
                            st.session_state["fuz_confirm_mode"], st.session_state["fuz_overlap_count"] = True, overlap_count
                            st.rerun()
                        else: should_execute = True
                
                if should_execute:
                    with st.spinner("Processing..."):
                        count_added = 0
                        # 1. Collect all overlaps for selected candidates
                        overlaps_to_reject = set()
                        for idx in selected_indices:
                            cand = page_candidates[idx]
                            s_key = f"fuz_start_{cand['sentence_id']}_{cand['start_char']}"
                            e_key = f"fuz_end_{cand['sentence_id']}_{cand['start_char']}"
                            c_start = st.session_state.get(s_key, cand['start_char'])
                            c_end = st.session_state.get(e_key, cand['end_char'])
                            overlaps = db.check_overlapping_annotations(cand['sentence_id'], c_start, c_end)
                            for ov in overlaps:
                                # Use (sentence_id, start_char, end_char) as unique key
                                overlaps_to_reject.add((cand['sentence_id'], ov['start_char'], ov['end_char']))
                        # 2. Reject all overlaps in a batch
                        for sid, s, e in overlaps_to_reject:
                            db.reject_overlapping_annotations(sid, s, e)
                        # 3. Accept selected candidates
                        for idx in selected_indices:
                            cand = page_candidates[idx]
                            s_key = f"fuz_start_{cand['sentence_id']}_{cand['start_char']}"
                            e_key = f"fuz_end_{cand['sentence_id']}_{cand['start_char']}"
                            c_start = st.session_state.get(s_key, cand['start_char'])
                            c_end = st.session_state.get(e_key, cand['end_char'])
                            db.execute_query(
                                "INSERT INTO annotations (sentence_id, text_span, label, start_char, end_char, confidence, source_agent, is_accepted, manual_edit_type, created_at) VALUES (?, ?, ?, ?, ?, 1.0, 'fuzzy_search', ?, 'fuzzy_accept', CURRENT_TIMESTAMP)",
                                (cand['sentence_id'], cand['sentence_text'][c_start:c_end], label, c_start, c_end, True)
                            )
                            count_added += 1
                        if 'fuzzy_candidates_cache' in st.session_state:
                            del st.session_state['fuzzy_candidates_cache']
                        st.toast(f"✅ Added {count_added} annotations!")
                        st.rerun()

                # Reject Bulk (new)
                if ba2.button(f"🚫 Reject Selected ({len(selected_indices)})", key="fuz_reject_bulk", type="primary", disabled=len(selected_indices)==0):
                    with st.spinner("Processing Rejection..."):
                        cursor = db.conn.cursor()
                        count_rejected = 0
                        try:
                            for idx in selected_indices:
                                cand = page_candidates[idx]
                                s_key = f"fuz_start_{cand['sentence_id']}_{cand['start_char']}"
                                e_key = f"fuz_end_{cand['sentence_id']}_{cand['start_char']}"
                                c_start = st.session_state.get(s_key, cand['start_char'])
                                c_end = st.session_state.get(e_key, cand['end_char'])
                                # Cleanup Session State
                                if s_key in st.session_state: del st.session_state[s_key]
                                if e_key in st.session_state: del st.session_state[e_key]
                                c_text = cand['sentence_text']
                                final_span = c_text[c_start:c_end]
                                old_label = cand.get('label')
                                target_label = label if (force_label_update or old_label == label) else old_label
                                manual_edit_type = "fuzzy_reject"
                                cursor.execute(
                                    f"UPDATE annotations SET is_rejected={true_val}, is_accepted={false_val}, label={ph}, start_char={ph}, end_char={ph}, text_span={ph}, manual_edit_type={ph}, created_at=CURRENT_TIMESTAMP WHERE id={ph}",
                                    (target_label, c_start, c_end, final_span, manual_edit_type, cand['id'])
                                )
                                # Cleanup Session State
                                if f"fuz_start_{cand['id']}" in st.session_state: del st.session_state[f"fuz_start_{cand['id']}"]
                                if f"fuz_end_{cand['id']}" in st.session_state: del st.session_state[f"fuz_end_{cand['id']}"]
                                count_rejected += 1
                            db.conn.commit()
                        except Exception as e:
                            db.conn.rollback()
                            st.error(f"Error during bulk reject: {e}")
                        st.toast(f"🚫 Rejected {count_rejected} annotations!")
                        # Invalidate cache to force refresh
                        if 'fuzzy_candidates_cache' in st.session_state:
                            del st.session_state['fuzzy_candidates_cache']
                        st.rerun()

                if total_pages > 1:
                    pc1, pc2, pc3 = st.columns([1, 2, 1])
                    if pc1.button("⬅️ Prev", key="fuz_prev_btn", disabled=(st.session_state.fuzzy_page == 0)):
                        st.session_state.fuzzy_page -= 1
                        st.rerun()
                    pc2.markdown(f"<div style='text-align:center'>Page {st.session_state.fuzzy_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
                    if pc3.button("Next ➡️", key="fuz_next_btn", disabled=(st.session_state.fuzzy_page >= total_pages - 1)):
                        st.session_state.fuzzy_page += 1
                        st.rerun()

render_fuzzy_search_interface()






