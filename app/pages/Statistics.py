import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import importlib
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(
    page_title="Annotation Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Annotation Analytics Dashboard")
st.button("🔄 Refresh Data", on_click=st.rerun)

# --- Sidebar Filters ---
st.sidebar.header("Filters")
dataset_split = st.sidebar.selectbox("Dataset Split", ["train", "test", "validation", "All"], index=0)

# --- Database Connection (Reused from app.py) ---
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

@st.cache_resource(ttl=3600, validate=validate_db_connection)
def get_db_manager():
    # Check for DATABASE_URL in Env or Secrets
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        try:
            if "DATABASE_URL" in st.secrets:
                db_url = st.secrets["DATABASE_URL"]
            elif "postgres" in st.secrets and "url" in st.secrets["postgres"]:
                db_url = st.secrets["postgres"]["url"]
            elif "supabase" in st.secrets and "url" in st.secrets["supabase"]:
                db_url = st.secrets["supabase"]["url"]
        except Exception:
            pass

    if db_url:
        os.environ["DATABASE_URL"] = db_url
        if 'src.database.db_manager_postgres' in sys.modules:
            importlib.reload(sys.modules['src.database.db_manager_postgres'])
        from src.database.db_manager_postgres import DBManager
        return DBManager(db_url=db_url)
    else:
        if 'src.database.db_manager' in sys.modules:
            importlib.reload(sys.modules['src.database.db_manager'])
        from src.database.db_manager import DBManager
        return DBManager()

try:
    db = get_db_manager()
except Exception as e:
    st.error(f"Database Connection Error: {e}")
    st.stop()

# --- Helper Functions ---
def run_query(query, params=None):
    if params is None:
        params = ()
    
    try:
        # Check if it's Postgres (which uses RealDictCursor by default in this app)
        if "psycopg2" in str(type(db.conn)):
            with db.conn.cursor() as cur:
                cur.execute(query, params)
                # Fetch as list of dicts (RealDictRow)
                data = cur.fetchall()
                if not data:
                    return pd.DataFrame()
                # Convert to DataFrame (handles dict keys as columns correctly)
                return pd.DataFrame([dict(row) for row in data])
        else:
            # SQLite (Standard)
            return pd.read_sql_query(query, db.conn, params=params)
    except Exception as e:
        # st.error(f"Query Error: {e}")
        return pd.DataFrame()

@st.dialog("ℹ️ Επεξήγηση Στατιστικών & Φίλτρων")
def show_help_modal():
    st.markdown("""
    ### 📊 Οδηγός Ανάλυσης Δεδομένων
    
    Αυτή η σελίδα προσφέρει μια επισκόπηση της προόδου και της ποιότητας των σχολιασμών.
    
    ---
    
    #### 1️⃣ Τύποι Επεξεργασίας (Manual Edit Type)
    Περιγράφει **πώς** κατέληξε μια οντότητα στην τρέχουσα κατάστασή της (Accepted/Rejected).
    
    - **manual_accept**: Ο χρήστης πάτησε το κουμπί ✅ (Accept) χωρίς αλλαγές.
    - **manual_reject**: Ο χρήστης πάτησε το κουμπί ❌ (Reject).
    - **manual_add**: Ο χρήστης δημιούργησε την οντότητα από το μηδέν (Add Missing).
    - **fuzzy_accept**: Η οντότητα βρέθηκε και προστέθηκε μέσω **Fuzzy Search** (Αναζήτηση στο κείμενο).
    - **memory_accept**: Η οντότητα προστέθηκε μαζικά μέσω **Memory Propagation** (Εξάπλωση Μνήμης).
    - **accepted_modified_boundaries**: Ο χρήστης άλλαξε τα όρια (Start/End) και μετά έκανε Accept.
    - **accepted_modified_label**: Ο χρήστης άλλαξε την κατηγορία (Label) και μετά έκανε Accept.
    - **auto_reject_overlap**: Το σύστημα απέρριψε αυτόματα την οντότητα επειδή κάλυπτε το ίδιο κείμενο με μια άλλη που έγινε Accept.

    ---

    #### 2️⃣ Τύποι Επίλυσης (Resolution Type)
    Περιγράφει την προέλευση της οντότητας από την **αυτόματη διαδικασία** (πριν τον ανθρώπινο έλεγχο).
    
    - **Direct**: Η οντότητα προτάθηκε από ένα μόνο μοντέλο ή από πολλά που συμφώνησαν απόλυτα, χωρίς συγκρούσεις.
    - **Score_Based**: Υπήρξε σύγκρουση (επικάλυψη) μεταξύ προτάσεων και επιλέχθηκε αυτή με το υψηλότερο σκορ εμπιστοσύνης.
    - **LLM_Conflict_Resolution**: Υπήρξε σοβαρή σύγκρουση (κοντινά σκορ) και το LLM κλήθηκε να επιλέξει τον νικητή.
    - **LLM_Ambiguity_Resolution**: Ειδική περίπτωση σύγκρουσης μεταξύ GPE και LOCATION που λύθηκε από το LLM.
    - **LLM_GrayZone_Refinement**: Η οντότητα είχε χαμηλό σκορ (Gray Zone) και το LLM την επιβεβαίωσε ή τη διόρθωσε.
    - **LLM_Boundary_Refinement**: Το LLM διόρθωσε τα όρια της οντότητας (π.χ. αφαίρεσε σημεία στίξης ή stopwords) επειδή φαινόταν "ύποπτη".

    ---

    #### 3️⃣ Άλλα Φίλτρα
    - **Agent Family:** Το βασικό μοντέλο που πρότεινε την οντότητα (π.χ. `RoBERTa`, `GPE_Regex`).
    - **Agent Proposal Type:** Ο τύπος που πρότεινε το μοντέλο (π.χ. `ORG`, `PERSON`).
    - **Confidence:** Ο βαθμός βεβαιότητας του μοντέλου (0.0 - 1.0).
    - **Min Entity Frequency:** Εμφανίζει μόνο οντότητες που υπάρχουν τουλάχιστον X φορές στη βάση.
    - **Flagged Only:** Εμφανίζει μόνο προτάσεις που έχουν επισημανθεί με σημαία (🚩).
    """)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")

if st.sidebar.button("ℹ️ Οδηγίες & Επεξηγήσεις", type="primary", use_container_width=True):
    show_help_modal()

# Date Range
min_date_query = "SELECT MIN(created_at) FROM annotations"
max_date_query = "SELECT MAX(created_at) FROM annotations"
try:
    min_date_str = run_query(min_date_query).iloc[0, 0]
    max_date_str = run_query(max_date_query).iloc[0, 0]
    
    if min_date_str and max_date_str:
        min_date = pd.to_datetime(min_date_str).date()
        max_date = pd.to_datetime(max_date_str).date()
    else:
        min_date = datetime.now().date() - timedelta(days=30)
        max_date = datetime.now().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
except Exception as e:
    st.sidebar.error(f"Could not load dates: {e}")
    date_range = []

# Labels
all_labels = run_query("SELECT DISTINCT label FROM annotations ORDER BY label")['label'].tolist()
selected_labels = st.sidebar.multiselect("Entity Labels", all_labels, default=all_labels)

# Status
status_options = ["Accepted", "Rejected", "Pending"] # Derived from is_accepted/is_rejected
selected_status = st.sidebar.multiselect("Status", status_options, default=["Accepted"])

# Min Confidence
min_confidence = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0, 0.01)

# Source Agents & Types Parsing
raw_agents = run_query("SELECT DISTINCT source_agent FROM annotations")
unique_agent_names = set()
unique_agent_types = set()

if not raw_agents.empty:
    for agents_str in raw_agents['source_agent'].dropna():
        for full_agent_str in agents_str.split(','):
            full_agent_str = full_agent_str.strip()
            if '_' in full_agent_str:
                # Split on first underscore only (e.g. LLM_Conflict_Resolution -> LLM, Conflict_Resolution)
                parts = full_agent_str.split('_', 1)
                if len(parts) == 2:
                    unique_agent_names.add(parts[0])
                    unique_agent_types.add(parts[1])
            else:
                # Fallback for no underscore
                unique_agent_names.add(full_agent_str)

# Agent Name Filter
selected_agent_names = st.sidebar.multiselect("Agent Family", sorted(list(unique_agent_names)), default=[])

# Agent Type Filter
selected_agent_types = st.sidebar.multiselect("Agent Proposal Type", sorted(list(unique_agent_types)), default=[])

# Resolution Type
raw_resolutions = run_query("SELECT DISTINCT resolution_type FROM annotations")
all_resolutions = []
if not raw_resolutions.empty:
    all_resolutions = raw_resolutions['resolution_type'].dropna().unique().tolist()
selected_resolutions = st.sidebar.multiselect("Resolution Type", sorted(all_resolutions), default=[])

# Flagged / Commented
flagged_only = st.sidebar.checkbox("🚩 Show Flagged Only", value=False)

# Manual Edit Type
raw_edit_types = run_query("SELECT DISTINCT manual_edit_type FROM annotations")
all_edit_types = []
if not raw_edit_types.empty:
    all_edit_types = raw_edit_types['manual_edit_type'].dropna().unique().tolist()
selected_edit_types = st.sidebar.multiselect("Manual Edit Type", sorted(all_edit_types), default=[])

# Min Entity Frequency
# Calculate max frequency first to set slider range
max_freq_query = """
    SELECT MAX(cnt) 
    FROM (
        SELECT COUNT(*) as cnt 
        FROM annotations 
        GROUP BY text_span, label
    ) as sub
"""
try:
    max_freq = run_query(max_freq_query).iloc[0, 0]
    if pd.isna(max_freq): max_freq = 1
except:
    max_freq = 10

min_freq = st.sidebar.slider("Min Entity Frequency", 1, int(max_freq), 1, help="Show only entities that appear at least X times.")

# --- Data Fetching ---

# Construct WHERE clause
where_clauses = []
params = []

# Date Filter
if len(date_range) == 2:
    where_clauses.append("created_at >= %s AND created_at <= %s")
    # Adjust for end of day
    params.extend([date_range[0], date_range[1] + timedelta(days=1)])
elif len(date_range) == 1:
    where_clauses.append("created_at >= %s")
    params.append(date_range[0])

# Label Filter
if selected_labels:
    placeholders = ','.join(['%s'] * len(selected_labels))
    where_clauses.append(f"label IN ({placeholders})")
    params.extend(selected_labels)

# Min Confidence Filter
if min_confidence > 0:
    where_clauses.append("confidence >= %s")
    params.append(min_confidence)

# Source Agent Logic (Combinatorial)
agent_conditions = []

# Case 1: Both Filters Active (Combinations)
if selected_agent_names and selected_agent_types:
    for name in selected_agent_names:
        for type_ in selected_agent_types:
            # Construct target string e.g. "RoBERTa_ORG"
            target = f"{name}_{type_}"
            agent_conditions.append("source_agent LIKE %s")
            params.append(f"%{target}%")

# Case 2: Only Agent Names Active
elif selected_agent_names:
    for name in selected_agent_names:
        agent_conditions.append("source_agent LIKE %s")
        params.append(f"%{name}_%") # Expecting underscore after name

# Case 3: Only Agent Types Active
elif selected_agent_types:
    for type_ in selected_agent_types:
        agent_conditions.append("source_agent LIKE %s")
        params.append(f"%_{type_}%") # Expecting underscore before type

if agent_conditions:
    where_clauses.append(f"({' OR '.join(agent_conditions)})")

# Resolution Type Filter
if selected_resolutions:
    placeholders = ','.join(['%s'] * len(selected_resolutions))
    where_clauses.append(f"resolution_type IN ({placeholders})")
    params.extend(selected_resolutions)

# Manual Edit Type Filter
if selected_edit_types:
    placeholders = ','.join(['%s'] * len(selected_edit_types))
    where_clauses.append(f"manual_edit_type IN ({placeholders})")
    params.extend(selected_edit_types)

# Status Filter logic
is_postgres = "psycopg2" in str(type(db.conn))
status_conditions = []

if "Accepted" in selected_status:
    if is_postgres:
        status_conditions.append("is_accepted IS TRUE")
    else:
        status_conditions.append("is_accepted = 1")

if "Rejected" in selected_status:
    if is_postgres:
        status_conditions.append("is_rejected IS TRUE")
    else:
        status_conditions.append("is_rejected = 1")

if "Pending" in selected_status:
    if is_postgres:
        status_conditions.append("(is_accepted IS NOT TRUE AND is_rejected IS NOT TRUE)")
    else:
        status_conditions.append("(is_accepted = 0 AND is_rejected = 0)")

if status_conditions:
    where_clauses.append(f"({' OR '.join(status_conditions)})")

# Flagged / Commented Filter
# Note: Statistics page queries 'annotations' table, but flags are in 'sentences' table.
# We need to JOIN with sentences if this filter is active.
# However, the current query structure in Statistics.py is mostly on 'annotations'.
# Let's check if we can JOIN.
# The main queries are:
# count_query = f"SELECT COUNT(*) as count FROM annotations WHERE {where_sql}"
# entity_query = f"SELECT COUNT(DISTINCT text_span) FROM annotations WHERE {where_sql}"
# ...
# If we add a JOIN to the FROM clause, we need to change all queries.
# Alternatively, we can use a subquery in WHERE clause:
# sentence_id IN (SELECT id FROM sentences WHERE is_flagged = 1 OR comments IS NOT NULL)

if flagged_only:
    if is_postgres:
        where_clauses.append("sentence_id IN (SELECT id FROM sentences WHERE is_flagged IS TRUE)")
    else:
        where_clauses.append("sentence_id IN (SELECT id FROM sentences WHERE is_flagged = 1)")

# Min Frequency Filter (Subquery)
if min_freq > 1:
    # We need to filter annotations where the (text_span, label) pair appears >= min_freq times
    # This is a bit heavy but necessary for correct filtering
    freq_subquery = f"""
        text_span IN (
            SELECT text_span 
            FROM annotations 
            GROUP BY text_span, label 
            HAVING COUNT(*) >= {min_freq}
        )
    """
    where_clauses.append(freq_subquery)

where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

# Fix params for sqlite vs postgres (simple hack: replace %s with ? if sqlite)
if not is_postgres:
    where_sql = where_sql.replace("%s", "?")

# --- KPI Metrics ---
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

def safe_get_scalar(df, default=0):
    if not df.empty:
        return df.iloc[0, 0]
    return default

# Total Annotations
count_query = f"SELECT COUNT(*) as count FROM annotations WHERE {where_sql}"
total_annotations = safe_get_scalar(run_query(count_query, params))
col1.metric("Total Annotations", f"{total_annotations:,}")

# Unique Entities
entity_query = f"SELECT COUNT(DISTINCT text_span) FROM annotations WHERE {where_sql}"
unique_entities = safe_get_scalar(run_query(entity_query, params))
col2.metric("Unique Entities", f"{unique_entities:,}")

# Avg Confidence
conf_query = f"SELECT AVG(confidence) FROM annotations WHERE {where_sql}"
avg_conf = safe_get_scalar(run_query(conf_query, params), default=None)
col3.metric("Avg. Confidence", f"{avg_conf:.2%}" if avg_conf else "N/A")

# Total Sentences (Global)
total_sentences = safe_get_scalar(run_query("SELECT COUNT(*) FROM sentences"))
col4.metric("Total Sentences in DB", f"{total_sentences:,}")

st.divider()

# --- Charts ---
st.subheader("📊 Visualizations")

# Row 1: Label Distribution & Source Agents
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.markdown("##### 🏷️ Annotations by Label & Status")
    # Fetch data for Labels + Status
    # We fetch raw columns and process in Pandas to handle DB differences (bool vs int) safely
    lbl_stat_query = f"SELECT label, is_accepted, is_rejected FROM annotations WHERE {where_sql}"
    df_lbl_stat = run_query(lbl_stat_query, params)
    
    if not df_lbl_stat.empty:
        # Determine status
        def get_status(row):
            # Handle both boolean (Postgres) and 1/0 (SQLite)
            acc = row['is_accepted'] == 1 or row['is_accepted'] is True
            rej = row['is_rejected'] == 1 or row['is_rejected'] is True
            if acc: return 'Accepted'
            if rej: return 'Rejected'
            return 'Pending'
            
        df_lbl_stat['status'] = df_lbl_stat.apply(get_status, axis=1)
        
        # Group
        df_grouped = df_lbl_stat.groupby(['label', 'status']).size().reset_index(name='count')
        
        # Chart
        fig_labels = px.bar(
            df_grouped, 
            x='label', 
            y='count', 
            color='status',
            color_discrete_map={'Accepted': '#00CC96', 'Rejected': '#EF553B', 'Pending': '#FECB52'},
            title="Labels by Status",
            barmode='stack'
        )
        st.plotly_chart(fig_labels, use_container_width=True)
    else:
        st.info("No data for labels.")

with r1c2:
    st.markdown("##### 🤖 Top Source Agents")
    # Group by source_agent
    agent_query = f"SELECT source_agent, COUNT(*) as count FROM annotations WHERE {where_sql} GROUP BY source_agent ORDER BY count DESC LIMIT 15"
    df_agents = run_query(agent_query, params)
    
    if not df_agents.empty:
        fig_agents = px.bar(
            df_agents, 
            x='count', 
            y='source_agent', 
            orientation='h',
            title="Annotations per Source Agent",
            color='count',
            color_continuous_scale='Viridis'
        )
        fig_agents.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_agents, use_container_width=True)
    else:
        st.info("No agent data available.")

# Row 2: Resolution Types & Confidence
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown("##### ⚖️ Resolution Types")
    res_query = f"SELECT resolution_type, COUNT(*) as count FROM annotations WHERE {where_sql} GROUP BY resolution_type"
    df_res = run_query(res_query, params)
    
    if not df_res.empty:
        # Clean up None/Null
        df_res['resolution_type'] = df_res['resolution_type'].fillna('Unknown')
        
        fig_res = px.pie(
            df_res, 
            names='resolution_type', 
            values='count', 
            hole=0.4,
            title="Conflict Resolution Strategy"
        )
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("No resolution data.")

with r2c2:
    st.markdown("##### 🎯 Confidence Distribution")
    conf_dist_query = f"SELECT confidence FROM annotations WHERE {where_sql} AND confidence IS NOT NULL"
    df_conf = run_query(conf_dist_query, params)
    
    if not df_conf.empty:
        fig_conf = px.histogram(
            df_conf, 
            x='confidence', 
            nbins=20, 
            title="Confidence Score Distribution",
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("No confidence data.")

# Row 3: Manual Edit Types (Existing but moved)
st.markdown("##### ✍️ Manual Edit Types")
source_query = f"SELECT manual_edit_type, COUNT(*) as count FROM annotations WHERE {where_sql} GROUP BY manual_edit_type"
df_source = run_query(source_query, params)
if not df_source.empty:
    # Clean None
    df_source['manual_edit_type'] = df_source['manual_edit_type'].fillna('N/A')
    
    fig_source = px.pie(
        df_source, 
        names='manual_edit_type', 
        values='count', 
        title="Manual Actions Breakdown"
    )
    st.plotly_chart(fig_source, use_container_width=True)
else:
    st.info("No manual edit data.")

# --- Annotation Explorer (Replaces Top Entities) ---
st.subheader("🗂️ Annotation Explorer")
st.info("Select a row to jump to that sentence in the main App.")

# 1. Fetch All Annotations (Filtered)
# We need to calculate frequency for each text_span to allow sorting by it
# Window functions are cleaner, but let's do it in Pandas to be safe across DBs
explorer_query = f"""
    SELECT id, text_span, label, sentence_id, confidence, is_accepted, is_rejected, created_at 
    FROM annotations 
    WHERE {where_sql}
"""
df_explorer = run_query(explorer_query, params)

if not df_explorer.empty:
    # Calculate Frequency
    freq_map = df_explorer.groupby(['text_span', 'label']).size().reset_index(name='frequency')
    df_explorer = pd.merge(df_explorer, freq_map, on=['text_span', 'label'], how='left')
    
    # Sort by Frequency DESC by default
    df_explorer = df_explorer.sort_values(by='frequency', ascending=False)
    
    # Reorder columns
    cols = ['frequency', 'text_span', 'label', 'sentence_id', 'confidence', 'is_accepted', 'created_at']
    df_display = df_explorer[cols].copy()
    
    # Interactive Table
    selection = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    # Handle Selection
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        selected_row = df_display.iloc[idx]
        target_sid = int(selected_row['sentence_id'])
        target_text = selected_row['text_span']
        
        st.success(f"Selected: **{target_text}** (Sentence #{target_sid})")
        
        if st.button(f"🚀 Jump to Sentence #{target_sid} in App"):
            st.session_state["target_sentence_id"] = target_sid
            st.switch_page("app.py")
else:
    st.warning("No annotations found matching the filters.")

# --- Raw Data View ---
with st.expander("📄 View Raw Data (Sample 100)"):
    raw_query = f"SELECT * FROM annotations WHERE {where_sql} ORDER BY created_at DESC LIMIT 100"
    df_raw = run_query(raw_query, params)
    st.dataframe(df_raw)
