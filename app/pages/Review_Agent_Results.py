import streamlit as st
import sqlite3
import pandas as pd
import json

# Configuration
DB_PATH = "data/production_annotations.db"

def init_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = init_connection()

st.set_page_config(page_title="Review AI Sweeper Results", layout="wide")

st.title("🕵️ Review AI Sweeper Results (Train Set)")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Filter by Status
filter_status = st.sidebar.radio("Annotation Status", ["Accepted (Discovery)", "Pending Review", "All"])

# Filter by Label
labels = pd.read_sql("SELECT DISTINCT label FROM annotations", conn)['label'].tolist()
selected_label = st.sidebar.selectbox("Filter by Label", ["All"] + labels)

# --- Fetch Data ---
query = """
    SELECT 
        a.id as ann_id,
        s.text as full_text,
        a.text_span,
        a.label,
        a.confidence,
        a.trigger_text
    FROM annotations a
    JOIN sentences s ON a.sentence_id = s.id
"""

conditions = []
params = []

if filter_status == "Accepted (Discovery)":
    conditions.append("a.trigger_text = 'Agent-Discovery'")
    conditions.append("a.is_accepted = 1")

if selected_label != "All":
    conditions.append("a.label = ?")
    params.append(selected_label)

if conditions:
    query += " WHERE " + " AND ".join(conditions)

query += " ORDER BY a.id DESC LIMIT 100"

try:
    df = pd.read_sql(query, conn, params=params)
    
    st.metric("Total Findings", len(df))
    
    if df.empty:
        st.info("No annotations found matching criteria.")
    else:

        for _, row in df.iterrows():
            with st.expander(f"ID {row['ann_id']} | {row['label']} | {row['text_span']}"):
                # Highlight logic
                text = row['full_text']
                span = row['text_span']
                
                parts = text.split(span)
                if len(parts) >= 2:
                    annotated_html = f"{parts[0]}<span style='background-color: #ffd700; padding: 2px; border-radius: 4px;'><b>{span}</b> ({row['label']})</span>{parts[1]}"
                    st.markdown(annotated_html, unsafe_allow_html=True)
                else:
                    st.write(text) # Fallback
                
                c1, c2 = st.columns(2)
                if c1.button("❌ Reject", key=f"rej_{row['ann_id']}"):
                    conn.execute("UPDATE annotations SET is_accepted=0, is_rejected=1 WHERE id=?", (row['ann_id'],))
                    conn.commit()
                    st.experimental_rerun()
                    
                if c2.button("✅ Confirm", key=f"conf_{row['ann_id']}"):
                    # Already confirmed effectively, but maybe we mark as valid validation?
                    st.success("Confirmed!")
                    
except Exception as e:
    st.error(f"Error: {e}")
