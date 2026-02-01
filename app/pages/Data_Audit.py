import streamlit as st
import pandas as pd
import sys
import os
import sqlite3
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.db_manager import DBManager
# from src.judges.llm_client import LLMJudge

# Init DB
DB_PATH = "data/production_annotations.db"
db = DBManager(db_path=DB_PATH)

st.set_page_config(page_title="Data Audit Tool", layout="wide")

st.title("🛡️ Data Audit & Corrections")

# --- Tabs ---
tab_recent, tab_missing, tab_conflicts = st.tabs(["🕒 Recent Activity (Rollback)", "🔍 Find Missing", "⚔️ Conflict Resolution"])

with tab_recent:
    st.header("🕒 Recent Sentences (Review & Wipe)")
    
    # 1. Stats
    try:
        total_approved = db.conn.execute("SELECT COUNT(*) FROM sentences WHERE status='approved'").fetchone()[0]
        total_anns = db.conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        st.metric("Total Completed Sentences", total_approved, delta=f"{total_anns} annotations")
    except Exception as e:
        total_approved = 0
        st.error(f"DB Error: {e}")

    # 2. Controls
    c_lim, c_sort = st.columns([1, 2])
    with c_lim:
        limit = st.slider("Show Last N Sentences", 10, 500, 50)
    
    # query: Get approved sentences, ordered by their latest annotation time
    query = f"""
        SELECT 
            s.id,
            s.text as Text,
            COUNT(a.id) as Annotation_Count,
            MAX(a.created_at) as Last_Modified
        FROM sentences s
        LEFT JOIN annotations a ON s.id = a.sentence_id
        WHERE s.status = 'approved'
        GROUP BY s.id
        ORDER BY Last_Modified DESC
        LIMIT {limit}
    """
    
    df = pd.read_sql(query, db.conn)
    
    if not df.empty:
        st.write("### 📋 Recently Completed Sentences")
        
        # Controls for Selection
        c_sel_all, c_desel_all, c_filler = st.columns([1, 1, 3])
        
        # Session state to track "Select All" toggle
        if "audit_select_all_state" not in st.session_state:
            st.session_state.audit_select_all_state = False
            
        with c_sel_all:
            if st.button("Select All Visible"):
                st.session_state.audit_select_all_state = True
                st.rerun()
        
        with c_desel_all:
             if st.button("Deselect All"):
                st.session_state.audit_select_all_state = False
                st.rerun()

        # Add 'Reset' Checkbox
        df['Reset'] = st.session_state.audit_select_all_state
        
        # Reorder
        df = df[['Reset', 'id', 'Text', 'Annotation_Count', 'Last_Modified']]
        
        # Use dynamic key to force refresh when Select All changes
        table_key = f"sentence_audit_table_{st.session_state.audit_select_all_state}_{limit}"
        
        edited_df = st.data_editor(
            df,
            key=table_key,
            hide_index=True,
            column_config={
                "Reset": st.column_config.CheckboxColumn("Select", help="Select to wipe and re-do this sentence"),
                "Text": st.column_config.TextColumn("Sentence Text", width="large"),
                "Last_Modified": st.column_config.DatetimeColumn(format="D MMM, HH:mm")
            },
            disabled=["id", "Text", "Annotation_Count", "Last_Modified"]
        )
        
        # Actions
        to_reset = edited_df[edited_df['Reset'] == True]
        
        if not to_reset.empty:
            st.warning(f"⚠️ You have selected {len(to_reset)} sentences.")
            
            col_reset, col_delete = st.columns(2)
            
            # OPTION 1: SOFT RESET (Re-Annotate)
            with col_reset:
                if st.button(f"♻️ RESET & RE-PREDICT", type="primary", help="Deletes current annotations and sends the sentence back to the queue. The CURRENT Model will generate NEW predictions when you open it."):
                    ids_to_wip = to_reset['id'].tolist()
                    placeholders = ','.join('?' * len(ids_to_wip))
                    
                    # 1. Reset Status to 'pending'
                    db.conn.execute(f"UPDATE sentences SET status='pending' WHERE id IN ({placeholders})", ids_to_wip)
                    
                    # 2. Delete ALL annotations for these sentences
                    db.conn.execute(f"DELETE FROM annotations WHERE sentence_id IN ({placeholders})", ids_to_wip)
                    
                    db.conn.commit()
                    st.success(f"Reset {len(ids_to_wip)} sentences! Go to 'Annotate' to see fresh predictions from the active model.")
                    time.sleep(1)
                    st.rerun()

            # OPTION 2: HARD DELETE (Remove Completely)
            with col_delete:
                if st.button(f"🔥 DELETE FROM DB", type="secondary", help="Permanently delete Sentence AND Annotations. (Use if sentence itself is bad)"):
                    ids_to_del = to_reset['id'].tolist()
                    placeholders = ','.join('?' * len(ids_to_del))
                    
                    # 1. Delete Annotations
                    db.conn.execute(f"DELETE FROM annotations WHERE sentence_id IN ({placeholders})", ids_to_del)
                    
                    # 2. Delete Sentences
                    db.conn.execute(f"DELETE FROM sentences WHERE id IN ({placeholders})", ids_to_del)
                    
                    db.conn.commit()
                    st.error(f"Permanently Deleted {len(ids_to_del)} sentences.")
                    time.sleep(1)
                    st.rerun()
                
    else:
        st.info("No approved sentences found yet.")

with tab_missing:
    st.subheader("Automatic Missing Entity Detection")
    st.write("Automatically scans sentences with NO annotations to find missed entities.")
    
    if st.button("🚀 Scan 10 Random Un-Annotated Sentences"):
        # 1. Fetch sentences with 0 annotations
        query = """
            SELECT s.id, s.text 
            FROM sentences s
            LEFT JOIN annotations a ON s.id = a.sentence_id
            WHERE a.id IS NULL
            AND s.dataset_split = 'train'
            ORDER BY RANDOM() LIMIT 10
        """
        results = db.conn.execute(query).fetchall()
        
        results_container = st.container()
        
        for row in results:
            sent_id = row['id']
            text = row['text']
            
            with st.expander(f"Sentence #{sent_id}", expanded=True):
                st.write(f"📝 **Text:** {text}")
                
                # Check with LLM
                with st.spinner("Asking LLM..."):
                    missing = st.session_state.llm_judge.find_missing_entities(text, [])
                
                if missing:
                    st.success(f"Found {len(missing)} potential entities!")
                    for m in missing:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.code(f"{m['text']} -> {m['label']}")
                        with col2:
                            st.caption(f"Reason: {m.get('reason', '')}")
                        with col3:
                             if st.button("Add", key=f"add_{sent_id}_{m['text']}"):
                                 # Logic to add to DB (Simplified)
                                 # We need to find char offsets
                                 start = text.find(m['text'])
                                 if start != -1:
                                     end = start + len(m['text'])
                                     db.add_annotation(sent_id, m['text'], m['label'], start, end, 
                                                       source="Audit_LLM", confidence=0.8, is_accepted=False)
                                     st.toast("Added!")
                                     st.rerun()
                                 else:
                                     st.error("Could not locate text exactly.")
                else:
                    st.info("No missing entities found.")
                    # Mark as "Audited_Empty" so we don't scan again?
                    # db.conn.execute("UPDATE sentences SET status='audited_empty' WHERE id=?", (sent_id,))
                    # db.conn.commit()

with tab_conflicts:
    st.subheader("Resolve Disagreements (Regex vs Model)")
    # Logic to fetch sentences where different sources give different labels for overlapping spans
    st.info("Under Construction: Will show sentences where Regex and RoBERTa disagree.")
