import threading
import time
import queue
import streamlit as st
import json
from src.judges.llm_client import LLMJudge
from src.database.db_manager import DBManager

# Singleton mechanism for Streamlit
@st.cache_resource
def get_background_fixer():
    return BackgroundFixer()

class BackgroundFixer:
    """
    Manages a background thread that processes 'Bounds Check' requests via LLM.
    Results are stored in a thread-safe queue and can be polled by the Streamlit app.
    """
    def __init__(self):
        self.input_queue = queue.Queue()
        self.completed_fixes = [] # List of dicts: {id, sentence_id, text_span, new_span, status}
        self.is_running = True
        self.llm_client = LLMJudge()
        self.db_manager = None # Will be set dynamically or use a new instance
        
        # Start Thread
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print("🔧 BackgroundFixer Thread Started")

    def _worker_loop(self):
        """
        Consumes tasks from input_queue and processes them.
        """
        while self.is_running:
            try:
                # Wait for a task
                task = self.input_queue.get(timeout=2.0) # 2 sec timeout
                
                # Execute Logic
                self._process_task(task)
                
                # Mark done
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in BackgroundFixer loop: {e}")

    def _process_task(self, task):
        """
        Calls LLM to refine boundaries OR find missing entities.
        """
        print(f"🔧 Processing Task: {task.get('type')} - {task.get('annotation_id', 'N/A')}")
        
        try:
            # --- TASK TYPE 1: BOUNDS FIX ---
            if task['type'] == 'bounds':
                refined_span = self.llm_client.refine_boundaries(
                    text_context=task['text_context'],
                    entity_text=task['current_span'],
                    label=task['label'],
                    user_feedback=True
                )
                
                if refined_span and refined_span != task['current_span']:
                    result = {
                        'type': 'bounds',
                        'original': task,
                        'refined_span': refined_span,
                        'status': 'fixed',
                        'timestamp': time.time()
                    }
                    self.completed_fixes.insert(0, result) 
                    print(f"✅ Fix Found: {refined_span}")

            # --- TASK TYPE 2: MISSING ENTITIES SCAN ---
            elif task['type'] == 'missing':
                found_entities = self.llm_client.find_missing_entities(
                    text=task['text_context'],
                    existing_entities=task['existing_entities']
                )
                
                if found_entities and len(found_entities) > 0:
                    result = {
                        'type': 'missing',
                        'original': task, # Has 'sentence_id'
                        'found': found_entities,
                        'status': 'found',
                        'timestamp': time.time()
                    }
                    self.completed_fixes.insert(0, result)
                    print(f"✅ Missing Entities Found: {len(found_entities)}")
                else:
                    print("✅ No missing entities found.")

        except Exception as e:
            print(f"❌ Background Task Failed: {e}")

    def add_task(self, task):
        self.input_queue.put(task)

    def get_pending_count(self):
        return self.input_queue.qsize()
    
    def get_completed_fixes(self):
        return self.completed_fixes

    def pop_latest_fix(self):
        if self.completed_fixes:
            return self.completed_fixes.pop(0)
        return None

    def clear_all(self):
        """Clears all completed fixes and attempts to drain the input queue."""
        self.completed_fixes = []
        # Drain queue
        try:
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
                self.input_queue.task_done()
        except queue.Empty:
            pass
        print("🧹 Background Fixer Queue Cleared")
