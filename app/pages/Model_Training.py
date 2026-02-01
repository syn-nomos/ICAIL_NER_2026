import streamlit as st
import os
import sys
import subprocess
import signal
import time

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.dataset_exporter import export_accepted_to_conll

st.set_page_config(page_title="Train Model", layout="wide")

st.title("🧠 Neural Network Training Center")
st.markdown("""
Here you can retrain the RoBERTa model using the latest accepted annotations from the database.
This process happens in two steps:
1. **Export** the database content to a standard CoNLL format.
2. **Train** the model using the exported file.
""")

# --- Constants ---
DB_PATH = "data/production_annotations.db"
DEFAULT_EXPORT_PATH = "data/retrain_dataset.conll"
DEFAULT_MODEL_OUTPUT = "src/agents/GreekLegalRoBERTa_New"
PID_FILE = "training.pid"
LOG_FILE = "training_output.log"

# --- Helper Functions ---

def is_training_running():
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        # Check if process exists (Windows)
        # On Windows, os.kill(pid, 0) works to check existence
        os.kill(pid, 0)
        return pid
    except OSError:
        return False
    except ValueError:
        return False

def stop_training():
    pid = is_training_running()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM) # Or SIGKILL
            st.success("Training process stopped.")
        except Exception as e:
            st.error(f"Error stopping process: {e}")
        finally:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
    else:
        st.warning("No active training process found.")

# --- UI Sections ---

col1, col2 = st.columns(2)

with col1:
    st.header("1. Export Data")
    st.info(f"Source Database: `{DB_PATH}`")
    export_path = st.text_input("Export Path", DEFAULT_EXPORT_PATH)
    
    if st.button("📤 Export Annotations to CoNLL"):
        if not os.path.exists(DB_PATH):
            st.error(f"Database not found at {DB_PATH}")
        else:
            with st.spinner("Exporting data..."):
                success, msg = export_accepted_to_conll(DB_PATH, export_path)
                if success:
                    st.success(msg)
                    st.session_state['data_ready'] = True
                else:
                    st.error(msg)

with col2:
    st.header("2. Train Model")
    
    # Check status
    pid = is_training_running()
    if pid:
        st.warning(f"⚠️ Training is currently running (PID: {pid})")
        if st.button("🛑 Stop Training"):
            stop_training()
            st.rerun()
            
        # Log Reader
        st.subheader("Live Logs")
        log_container = st.empty()
        
        if os.path.exists(LOG_FILE):
             with open(LOG_FILE, "r") as f:
                 lines = f.readlines()
                 log_container.code("".join(lines[-30:])) 
        else:
            log_container.write("Waiting for logs...")
            
        # Auto-refresh mechanism
        time.sleep(2)
        st.rerun()
            
    else:
        st.success("System Idle. Ready to train.")
        
        # Default to the Stable V3 model, OR fall back to Original Local Baseline
        # Lifecycle: Original -> V3 -> New
        
        candidates_base = [
            "src/agents/GreekLegalRoBERTa_v3",    # Preferred (Stable)
            "src/agents/Roberta_Base_Api/model",  # Backup (Original Local)
            "xlm-roberta-base"                    # Last resort (Huggingface)
        ]
        
        default_base = "xlm-roberta-base"
        for c in candidates_base:
            if os.path.exists(c) and (os.path.isdir(c) and "config.json" in os.listdir(c)):
                 default_base = c
                 break
            elif c == "xlm-roberta-base":
                 # If we reached here, use the string name
                 default_base = c

        base_model = st.text_input("Base Model (Starting Point)", default_base, help="The model to start training FROM.")
        output_dir = st.text_input("Output Directory (Result)", DEFAULT_MODEL_OUTPUT, help="Where to save the NEW model.")
        epochs = st.number_input("Epochs", min_value=1, max_value=20, value=3)
        input_file = st.text_input("Input Data File", export_path)
        
        if st.button("🚀 Start Training"):
            if not os.path.exists(input_file):
                st.error(f"Input file not found: {input_file}. Please export data first.")
            else:
                # Start Subprocess
                # Added -u for unbuffered output so logs show up immediately
                cmd = [
                    sys.executable, "-u", "train_new_roberta.py",
                    "--input_file", input_file,
                    "--output_dir", output_dir,
                    "--base_model", base_model,
                    "--epochs", str(epochs)
                ]
                
                try:
                    # Open Log File
                    log_f = open(LOG_FILE, "w")
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_f,
                        stderr=log_f,
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
                    )
                    
                    # Save PID
                    with open(PID_FILE, "w") as f:
                        f.write(str(proc.pid))
                        
                    st.toast(f"Training started! PID: {proc.pid}")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to start training: {e}")

st.markdown("---")
st.markdown("### Guide")
st.markdown("""
- **Export**: Converts your verified database annotations into a format the model understands.
- **Train**: Fine-tunes the RoBERTa model. Be patient, this can take minutes to hours depending on data size and hardware (GPU/CPU).
- **Reload**: After training, you must restart the main application to load the new model from the output directory.
""")
