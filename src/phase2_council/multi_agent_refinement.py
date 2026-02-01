import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judges.llm_client import LLMJudge
import sqlite3
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DB_PATH = "data/production_annotations.db"
MODEL_NAME = "deepseek/deepseek-v3.2" # Optimized for Speed & Agentic Performance
RE_DATE = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")

# ENTITY GUIDELINES (Extracted/Summarized for Greek Context)
GUIDELINES = {
    "LEG_REFS": """
    Αναζητάμε ΑΝΑΦΟΡΕΣ ΣΕ ΝΟΜΟΘΕΣΙΑ (Legal References).
    Συμπεριλαμβάνονται:
    - Νόμοι (Ν. 1234/2000, Νόμος υπ' αριθμ. 5)
    - Προεδρικά Διατάγματα (Π.Δ. 50/2001, π.δ/τος 5)
    - Υπουργικές Αποφάσεις (Υ.Α. 12345/2020)
    - Κανονισμοί/Οδηγίες ΕΕ (Κανονισμός (ΕΚ) 100/2000)
    - Κώδικες (Αστικός Κώδικας, Ποινικός Κώδικας, ΚΠολΔ)
    
    ΠΡΟΣΟΧΗ ΣΤΑ ΟΡΙΑ:
    - Να συμπεριλαμβάνεται ο αριθμός (π.χ. "Ν. 1234/90" ΟΧΙ σκέτο "Ν. 1234")
    - Αν υπάρχει τίτλος σε εισαγωγικά κολλητά, συμπεριέλαβε τον ΜΟΝΟ αν είναι μέρος της επίσημης ονομασίας. Συνήθως κρατάμε το τυπικό "Ν. 1234/2000".
    - ΜΗΝ συμπεριλαμβάνεις το "ΦΕΚ 123" εκτός αν είναι μέρος της ονομασίας του νόμου που δεν έχει άλλο αριθμό.
    - ΜΗΝ συμπεριλαμβάνεις "Άρθρο 5", "Παράγραφος 2". Θέλουμε ΜΟΝΟ την πράξη (π.χ. "Ν. 4953/1931"), ΟΧΙ "άρθρου 14 του Ν. 4953/1931".
    """,
    "GPE": """
    Geopolitical Entities (Γεωπολιτικές Οντότητες - Διοικητικές Περιοχές με Θεσμικό Ρόλο).
    - ***ΒΑΣΙΚΗ ΔΙΑΚΡΙΣΗ (GPE vs LOCATION)***:
      - Αν η οντότητα ΔΡΑ, ΑΠΟΦΑΣΙΖΕΙ, ΝΟΜΟΘΕΤΕΙ ή έχει ΝΟΜΙΚΗ ΠΡΟΣΩΠΙΚΟΤΗΤΑ -> GPE.
        - π.χ. "Η Ελλάδα υπέγραψε" (GPE).
        - π.χ. "Ο Δήμος αποφάσισε" (GPE).
        - π.χ. "Το Ελληνικό Δημόσιο" (GPE).
      - Αν η οντότητα είναι απλά γεωγραφικός προσδιορισμός, τόπος γέννησης ή κατοικίας -> LOCATION.
    
    - Περιλαμβάνει: Χώρες, Πόλεις, Δήμους, Περιφέρειες (όταν δρουν ως αρχές).
    """,
    "ORG": """
    Organizations (Οργανισμοί, Εταιρείες, Φορείς).
    - Υπουργεία (π.χ. "Υπουργείο Οικονομικών").
    - Αποκεντρωμένες Διοικήσεις, Περιφέρειες, Δήμοι (ως θεσμοί).
    - Τράπεζες, Εταιρείες, Δήμοι, Επιτροπές.
    
    *** ΑΥΣΤΗΡΟΣ ΚΑΝΟΝΑΣ (ΤΙΤΛΟΙ ΑΞΙΩΜΑΤΟΥΧΩΝ) ***:
    - ΑΠΑΓΟΡΕΥΕΤΑΙ να συμπεριλαμβάνεις τον τίτλο του προσώπου στο span του Οργανισμού.
    - Πρέπει να αφαιρείς τον τίτλο και να κρατάς ΜΟΝΟ το όνομα του Οργανισμού.
    
    ΠΑΡΑΔΕΙΓΜΑΤΑ ΛΑΘΩΝ vs ΣΩΣΤΩΝ:
    - ❌ ΛΑΘΟΣ: "Γενικός Γραμματέας Αποκεντρωμένης Διοίκησης Μακεδονίας" (Αυτό είναι Τίτλος).
    - ✅ ΣΩΣΤΟ: "Αποκεντρωμένης Διοίκησης Μακεδονίας" (Κράτα μόνο τον φορέα).
    
    - ❌ ΛΑΘΟΣ: "Υπουργός Εσωτερικών", "Αναπληρωτής Υπουργός", "Πρωθυπουργός".
    - ✅ ΣΩΣΤΟ: ΑΓΝΟΗΣΕ ΤΟ ΤΕΛΕΙΩΣ αν αναφέρεται στο πρόσωπο/ρόλο. Αν γράφει "Υπουργείο Εσωτερικών" πάρε το. Αν γράφει "Ο Υπουργός αποφασίζει", ΜΗΝ πάρεις το "Υπουργός" ως ORG.
    
    - ❌ ΛΑΘΟΣ: "Διοικητής Τράπεζας της Ελλάδος".
    - ✅ ΣΩΣΤΟ: "Τράπεζας της Ελλάδος".
    """,
    "DATE": """
    Ημερομηνίες.
    - Απόλυτες: 12 Ιανουαρίου 2020, 12/01/2020, 2005
    - Να αποφεύγονται ημερομηνίες που είναι μέρος αριθμών νόμων (εκτός αν είναι ξεκάθαρα ημερομηνία έκδοσης).
    """,
    "FACILITY": """
    Κτιριακές Εγκαταστάσεις / Υποδομές.
    - Αεροδρόμια, Λιμάνια, Φυλακές, Νοσοκομεία (ως κτίρια).
    - Οδοί/Δρόμοι (Οδός Πανεπιστημίου 10).
    """,
    "PUBLIC_DOCS": """
    Δημόσια Έγγραφα (που ΔΕΝ είναι Νόμοι).
    - Εγκύκλιοι, Προκηρύξεις, Διακηρύξεις, Συμβάσεις, Αριθμοί Πρωτοκόλλου.
    
    *** ΕΞΑΙΡΕΣΗ ΦΕΚ (FEK EXCLUSION) ***:
    - ΜΗΝ συμπεριλαμβάνεις αναφορές σε ΦΕΚ (π.χ. "ΦΕΚ 123/Α/2000") ή "Εφημερίδα της Κυβερνήσεως".
    - Θεωρούμε τα ΦΕΚ ως μεταδεδομένα/θόρυβο και ΟΧΙ ως οντότητες ενδιαφέροντος για αυτό το task.
    """,
    "PERSON": """
    Φυσικά Πρόσωπα.
    - ΜΟΝΟ ΤΟ ΟΝΟΜΑ (Όνομα, Επώνυμο, Πατρώνυμο).
    - π.χ. "Γ. Ανωμερίτης", "Ν. Χριστοδουλάκης".
    - ΑΠΑΓΟΡΕΥΕΤΑΙ η συμπερίληψη τίτλων/ιδιοτήτων μέσα στο span.
      - ΛΑΘΟΣ: "Υπουργό Γεωργίας Γ. Ανωμερίτη"
      - ΣΩΣΤΟ: "Γ. Ανωμερίτη"
      - ΣΩΣΤΟ: "Κ. Σημίτη του Γεωργίου" (επιτρέπεται το πατρώνυμο).
    - ***ΑΠΟΛΥΤΟΣ ΚΑΝΟΝΑΣ***: Λέξεις όπως 'Υπουργός', 'Πρόεδρος', 'Διοικητής', 'Διευθυντής' είναι ΤΙΤΛΟΙ. 
      - Αν το κείμενο είναι ΜΟΝΟ ο τίτλος (π.χ. 'ΥΠΟΥΡΓΟΣ ΕΘΝΙΚΗΣ ΠΑΙΔΕΙΑΣ'), ΤΟΤΕ ΑΓΝΟΗΣΕ ΤΟ ΤΕΛΕΙΩΣ. ΜΗΝ ΤΟ ΜΑΡΚΑΡΕΙΣ ΟΥΤΕ ΩΣ PERSON ΟΥΤΕ ΩΣ ORG.
      - Αν υπάρχει όνομα, κράτα ΜΟΝΟ το όνομα.
    """,
    "LOCATION": """
    Geographical Terms & Physical Locations (Γεωγραφικοί Όροι & Φυσικές Τοποθεσίες).
    - ***ΒΑΣΙΚΗ ΔΙΑΚΡΙΣΗ (GPE vs LOCATION)***:
      - Αν είναι απλά ΓΕΩΓΡΑΦΙΚΟΣ ΠΡΟΣΔΙΟΡΙΣΜΟΣ, Τόπος Γέννησης, Διεύθυνση -> LOCATION.
      - ΔΕΝ έχει διοικητική δράση εδώ. Είναι παθητικό στοιχείο (π.χ. "γεννήθηκε στην Αθήνα" -> LOCATION).
    - Περιλαμβάνει: Βουνά, Ποτάμια, Λίμνες, Νησίδες, Οδούς (π.χ. "Οδός Ερμού").
    - Περιλαμβάνει επίσης Πόλεις/Χωριά ΟΤΑΝ αναφέρονται απλά ως τοποθεσίες χωρίς δράση.
    """
}

class AutoAgent:
    def __init__(self):
        # Pass the configured MODEL_NAME explicitly
        self.llm = LLMJudge(model=MODEL_NAME)
        # self.conn removed to ensure thread safety (we create per-call connections)
        
    def get_pending_sentences_with_annotations(self, limit=100, offset=0):
        # Fetch pending sentences Randomly for better sampling
        # We start AFTER the cutoff (ID > 1762)
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.execute(f"""
                SELECT s.id, s.text 
                FROM sentences s
                WHERE s.status = 'pending' AND s.id > 1762
                ORDER BY RANDOM()
                LIMIT {limit}
            """)
            return c.fetchall()

    def get_annotations_for_sentence(self, sent_id, entity_type=None):
        query = "SELECT id, text_span, label, confidence, is_accepted FROM annotations WHERE sentence_id = ?"
        params = [sent_id]
        
        if entity_type:
             query += " AND label = ?"
             params.append(entity_type)
        
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.execute(query, tuple(params))
            return [dict(zip(['id', 'text', 'label', 'conf', 'accepted'], row)) for row in c.fetchall()]

    def get_few_shot_examples(self, label):
        """
        Retrieves real examples from the user's manual work to teach the Agent.
        Creates a new connection for thread safety.
        """
        examples = {"accepted": [], "rejected": [], "fixes": []}
        
        # Create thread-local connection
        local_conn = sqlite3.connect(DB_PATH)
        try:
            # 1. Accepted Examples (The "Gold Standard")
            c = local_conn.execute("""
                SELECT t FROM (
                    SELECT DISTINCT text AS t FROM allowlist WHERE label = ? 
                    UNION 
                    SELECT DISTINCT text_span AS t FROM annotations WHERE label = ? AND is_accepted = 1 
                ) ORDER BY LENGTH(t) ASC LIMIT 5
            """, (label, label))
            examples["accepted"] = [r[0] for r in c.fetchall()]
            
            # 2. Rejected Examples (What NOT to do)
            c = local_conn.execute("""
                SELECT DISTINCT text FROM denylist WHERE label = ? 
                UNION
                SELECT DISTINCT text_span FROM annotations WHERE label = ? AND is_accepted = 0 
                LIMIT 5
            """, (label, label))
            examples["rejected"] = [r[0] for r in c.fetchall()]
            
            # 3. Fix Rules (Boundary Corrections)
            c = local_conn.execute("""
                SELECT bad_text, good_text FROM fix_rules WHERE label = ? LIMIT 5
            """, (label,))
            examples["fixes"] = [{"bad": r[0], "good": r[1]} for r in c.fetchall()]
        finally:
            local_conn.close()
            
        return examples

    def run_specific_audit(self, entity_type, guidelines_text, sentence_id, text, current_anns):
        """
        Runs a specific auditor agent for ONE entity type.
        This provides a highly focused prompt to avoid confusion between similar entities.
        """
        if not current_anns: return None
        
        # 1. Fetch Dynamic Examples from DB
        history = self.get_few_shot_examples(entity_type)
        
        # Format Examples for Prompt
        ex_accepted = "\n".join([f"   - ✅ ΣΩΣΤΟ: '{t}'" for t in history['accepted']])
        ex_rejected = "\n".join([f"   - ❌ ΛΑΘΟΣ (Απόρριψη/Άλλο Τύπος): '{t}'" for t in history['rejected']])
        ex_fixes = "\n".join([f"   - 🔧 ΔΙΟΡΘΩΣΗ ΟΡΙΩΝ: '{f['bad']}' -> '{f['good']}'" for f in history['fixes']])
        
        # Prepare list for prompt
        anns_list_str = "\n".join([f"- ID {a['id']}: '{a['text']}'" for a in current_anns])
        
        prompt = f"""
        ΡΟΛΟΣ: Εξειδικευμένος Ελεγκτής για την οντότητα: [{entity_type}].
        ΓΛΩΣΣΑ: Ελληνικά (Νομικό Κείμενο).
        
        ΟΡΙΣΜΟΣ & ΟΔΗΓΙΕΣ ({entity_type}):
        {guidelines_text}
        
        --- ΠΑΡΑΔΕΙΓΜΑΤΑ ΑΠΟ ΤΗ ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ (USER HISTORY) ---
        Αυτά είναι πραγματικά παραδείγματα που ο χρήστης έχει ήδη ελέγξει. Μιμήσου τη λογική του.
        
        {ex_accepted if history['accepted'] else "   (Δεν υπάρχουν ακόμα αρκετά παραδείγματα)"}
        
        {ex_rejected if history['rejected'] else ""}
        
        {ex_fixes if history['fixes'] else ""}
        ---------------------------------------------------------
        
        ΣΤΟΧΟΣ: 
        Έλεγξε τις παρακάτω ΠΡΟΤΕΙΝΟΜΕΝΕΣ οντότητες που βρέθηκαν σε μια νέα πρόταση.
        Βεβαιώσου ότι ταιριάζουν στον ορισμό και στα παραδείγματα του χρήστη.
        
        ΠΡΟΤΑΣΗ: 
        "{text}"
        
        ΠΡΟΤΕΙΝΟΜΕΝΕΣ ΟΝΤΟΤΗΤΕΣ (Προς Έλεγχο):
        {anns_list_str}
        
        ΕΝΕΡΓΕΙΕΣ:
        1. CONFIRM (ΕΠΙΒΕΒΑΙΩΣΗ): Αν είναι σωστό ακριβώς όπως τα παραδείγματα.
        2. MODIFY (ΔΙΟΡΘΩΣΗ): Αν πιάνει παραπάνω κείμενο (π.χ. παρενθέσεις) ή λιγότερο. Δώσε το ακριβές σωστό κείμενο.
        3. DELETE (ΔΙΑΓΡΑΦΗ): Αν είναι λάθος τύπος (π.χ. Τοποθεσία που μαρκαρίστηκε ως Οργανισμός) ή θόρυβος.
        
        ΕΞΟΔΟΣ (JSON):
        {{
            "corrections": [
                {{ "id": <original_id>, "action": "CONFIRM" }},
                {{ "id": <original_id>, "action": "DELETE", "reason": "Not a {entity_type}" }},
                {{ "id": <original_id>, "action": "MODIFY", "text": "<exact_corrected_greek_text>" }}
            ]
        }}
        """
        # Call LLM ...
        try:
             # Using _call_llm (internal method as call_scan does not exist)
             response = self.llm._call_llm("You are an expert legal auditor.", prompt) 
             return response
        except Exception as e:
             print(f"Agent {entity_type} Error: {e}")
             return None

    def run_seeker_agent(self, agent_name, text, existing_labels):
        """
        Runs a 'Seeker' agent to find missing entities.
        """
        prompt = f"""
        ΡΟΛΟΣ: {agent_name} (Εμπειρογνώμονας Ανάλυσης Νομικού Κειμένου).
        ΓΛΩΣΣΑ: Ελληνικά.
        
        ΣΤΟΧΟΣ: 
        Σάρωσε το παρακάτω κείμενο και βρες ΕΓΚΥΡΕΣ οντότητες που ΔΕΝ υπάρχουν στη λίστα 'ΗΔΗ ΒΡΕΘΗΚΑΝ'.
        Πρέπει να λειτουργήσεις ανεξάρτητα και να εντοπίσεις οτιδήποτε έχει διαφύγει.
        
        ΟΡΙΣΜΟΙ ΟΝΤΟΤΗΤΩΝ (ΑΥΣΤΗΡΟΙ):
        {json.dumps(GUIDELINES, ensure_ascii=False, indent=2)}
        
        ΠΡΟΤΑΣΗ: "{text}"
        
        ΗΔΗ ΒΡΕΘΗΚΑΝ (Αγνόησε τα): {existing_labels}
        
        ΟΔΗΓΙΕΣ:
        1. Διάβασε την πρόταση πολύ προσεκτικά.
        2. Εντόπισε κάθε οντότητα που ταιριάζει στους ορισμούς και ΛΕΙΠΕΙ από τη λίστα.
        3. Προσοχή στα ΟΡΙΑ (Boundaries): Συμπερίλαβε ολόκληρους τους αριθμούς νόμων, ημερομηνίες, κτλ.
        4. Μην επινοείς οντότητες. Αν δεν είσαι σίγουρος, μην το βάλεις.
        5. ***ΑΠΑΓΟΡΕΥΣΗ ΕΓΚΙΒΩΤΙΣΜΟΥ (NO NESTED ENTITIES)***: Μην εξάγεις μια οντότητα αν είναι ήδη μέρος μιας άλλης μεγαλύτερης οντότητας (π.χ. "Κηφισιάς" μέσα στο "Αστυνομικό Τμήμα Κηφισιάς").
        
        ΕΞΟΔΟΣ JSON:
        {{ "found": [ {{ "text": "...", "label": "..." }} ] }}
        """
        try:
             # Calling scan (low temp) might produce identical results. 
             # In a real deployed version, we might vary temperature or prompt slightly.
             response = self.llm._call_llm("You are an expert legal entity seeker.", prompt) 
             return response
        except:
             return {"found": []}

    def run_consensus_judge(self, text, findings_a, findings_b, existing_labels):
        """
        The 'Third Agent' (Judge) that arbitrates between Seeker A and Seeker B.
        """
        prompt = f"""
        ΡΟΛΟΣ: Ανώτατος Δικαστής Σχολιασμού (Annotation Supreme Judge).
        ΓΛΩΣΣΑ: Ελληνικά.
        
        ΚΑΤΑΣΤΑΣΗ:
        Δύο ανεξάρτητοι ερευνητές (Ανιχνευτής Α και Ανιχνευτής Β) έψαξαν το κείμενο για οντότητες που ίσως έχουν χαθεί.
        Η δουλειά σου είναι να αποφασίσεις ποιες από τις προτάσεις τους είναι ΕΓΚΥΡΕΣ και να λύσεις διαφωνίες.
        
        ΠΡΟΤΑΣΗ: "{text}"
        
        ΕΥΡΗΜΑΤΑ ΕΡΕΥΝΗΤΗ Α: {json.dumps(findings_a, ensure_ascii=False)}
        ΕΥΡΗΜΑΤΑ ΕΡΕΥΝΗΤΗ Β: {json.dumps(findings_b, ensure_ascii=False)}
        
        ΗΔΗ ΥΠΑΡΧΟΝΤΑ (Context - Μην τα ξαναβγάλεις): {existing_labels}
        
        ΟΡΙΣΜΟΙ:
        {json.dumps(GUIDELINES, ensure_ascii=False, indent=2)}
        
        ΚΑΝΟΝΕΣ ΚΡΙΣΗΣ:
        1. ΣΥΜΦΩΝΙΑ (Ο Α και ο Β βρήκαν το ίδιο κείμενο/τύπο): -> ΑΥΤΟΜΑΤΗ ΑΠΟΔΟΧΗ (Είναι σίγουρα σωστό).
        2. ΣΥΓΚΡΟΥΣΗ ΟΡΙΩΝ (Ο Α λέει "Ν. 123", ο Β λέει "Ν. 123/2000"): -> ΕΠΙΛΕΞΕ ΤΟ ΠΙΟ ΠΛΗΡΕΣ και ΣΩΣΤΟ σύμφωνα με τους ορισμούς.
        3. ΔΙΑΦΩΝΙΑ (Το βρήκε μόνο ο ένας): -> ΑΞΙΟΛΟΓΗΣΕ ΑΥΣΤΗΡΑ. Αν είναι όντως οντότητα βάσει ορισμού, ΑΠΟΔΟΧΗ. Αν είναι θόρυβος, ΑΠΟΡΡΙΨΗ.
        4. ΕΓΚΙΒΩΤΙΣΜΟΣ (NESTED): Αν μια οντότητα περιέχεται ΟΛΟΚΛΗΡΗ μέσα σε μια άλλη (π.χ. 'Κηφισιά' GPE μέσα στο 'Αστυνομικό Τμήμα Κηφισιάς' ORG), ΚΡΑΤΑ ΜΟΝΟ ΤΗ ΜΕΓΑΛΥΤΕΡΗ (το ORG). Μην κρατάς και τα δύο.
        5. ΟΧΙ ΔΙΠΛΟΤΥΠΑ: Μην επιστρέψεις τίποτα που υπάρχει ήδη στη λίστα 'ΗΔΗ ΥΠΑΡΧΟΝΤΑ'.
        
        ΤΕΛΙΚΗ ΕΞΟΔΟΣ JSON:
        {{ "final_approved": [ {{ "text": "...", "label": "..." }} ] }}
        """
        try:
             response = self.llm._call_llm("You are a Supreme Legal Judge.", prompt) 
             return response
        except:
             return {"final_approved": []}

    def analyze_and_fix(self, sentence_id, text, current_anns):
        """
        Master orchestration function.
        """
        final_results = {'corrections': [], 'new_entities': []}
        
        # --- PHASE 1: SPECIFIC AUDITORS (Cleaning existing) ---
        anns_by_label = {}
        for a in current_anns:
            anns_by_label.setdefault(a['label'], []).append(a)
            
        # We track what remains valid to pass to Phase 2
        valid_current_spans = []
        
        # Helper for non-audited labels
        for label, anns in anns_by_label.items():
            if label not in GUIDELINES:
                 for a in anns: valid_current_spans.append(f"{a['text']} ({a['label']})")

        # SERIAL EXECUTION (To avoid Rate Limit Storms) 
        # Instead of launching 5 threads per sentence, we run auditors sequentially.
        # This prevents 8 sentences * 5 auditors = 40 concurrent requests locking the API.
        for label, anns in anns_by_label.items():
            if label in GUIDELINES:
                try:
                    res = self.run_specific_audit(label, GUIDELINES[label], sentence_id, text, anns)
                    
                    if res and 'corrections' in res:
                        final_results['corrections'].extend(res['corrections'])
                        
                        # Track what is confirmed/modified for context
                        for c in res['corrections']:
                            if c['action'] in ['CONFIRM', 'MODIFY']:
                                txt = c.get('text', next((x['text'] for x in anns if x['id']==c['id']), ''))
                                valid_current_spans.append(f"{txt} ({label})")
                except Exception as e:
                    print(f"Audit Error for {label}: {e}")

        # --- PHASE 2: DUAL DISCOVERY (Seeker A & B) ---
        # Run two independent passes in parallel (Safe to keep parallel, low overhead)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(self.run_seeker_agent, "Seeker A", text, valid_current_spans)
            future_b = executor.submit(self.run_seeker_agent, "Seeker B", text, valid_current_spans)
            
            res_a = future_a.result() or {"found": []}
            res_b = future_b.result() or {"found": []}
        
        findings_a = res_a.get('found', [])
        findings_b = res_b.get('found', [])
        
        # --- PHASE 3: THE JUDGE (Consensus) ---
        if findings_a or findings_b:
            res_judge = self.run_consensus_judge(text, findings_a, findings_b, valid_current_spans)
            if res_judge and 'final_approved' in res_judge:
                final_results['new_entities'].extend(res_judge['final_approved'])
            
        return final_results

    def extract_json(self, text):
        # Extract between ```Region
        m = re.search(r"```json(.*?)```", text, re.DOTALL)
        if m: return m.group(1)
        m = re.search(r"```(.*?)```", text, re.DOTALL)
        if m: return m.group(1)
        return text

    def apply_batch_fixes(self, sent_id, full_text, result):
        if not result: return
        
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            
            # 1. Corrections
            for fix in result.get('corrections', []):
                if fix['action'] == 'CONFIRM':
                    cur.execute("UPDATE annotations SET is_accepted=1, confidence=1.0 WHERE id=?", (fix['id'],))
                elif fix['action'] == 'DELETE':
                    cur.execute("DELETE FROM annotations WHERE id=?", (fix['id'],))
                elif fix['action'] == 'MODIFY':
                    new_txt = fix['text']
                    new_label = fix.get('label')
                    
                    # SAFETY CHECK: If label is missing (None), don't update label, keep existing?
                    # Or skip if critical. Let's try to fetch existing label if 'label' is missing from fix.
                    if not new_label:
                        # Fetch original label
                        orig = cur.execute("SELECT label FROM annotations WHERE id=?", (fix['id'],)).fetchone()
                        if orig: new_label = orig[0]
                        else: continue # Skip if we can't determine label
                        
                    # Search exact position to be safe (Python check)
                    if new_txt in full_text and new_label:
                        cur.execute("""
                            UPDATE annotations 
                            SET text_span=?, label=?, is_accepted=1, confidence=1.0, trigger_text='Agent-Fix'
                            WHERE id=?
                        """, (new_txt, new_label, fix['id']))
            
            # 2. New Entities
            for new in result.get('new_entities', []):
                txt = new['text']
                lbl = new['label']
                if txt in full_text:
                    cur.execute("""
                        INSERT INTO annotations (sentence_id, text_span, label, source_sentence, confidence, is_accepted, trigger_text)
                        VALUES (?, ?, ?, ?, 1.0, 1, 'Agent-Discovery')
                    """, (sent_id, txt, lbl, full_text))
                    
            # 3. Mark Sentence Approved?
            # User said "Once finished... we have completed the dataset".
            # So yes, we mark it approved.
            cur.execute("UPDATE sentences SET status='approved' WHERE id=?", (sent_id,))
            conn.commit()

def process_single_sentence(agent, sent_id, text):
    """
    Wrapper to process a single sentence in a thread.
    Returns a string log of what happened to avoid messy interleaved stdout.
    """
    # 0. SPEED & NOISE FILTER
    # Ελέγχουμε αν το κείμενο είναι πολύ μικρό ή "θόρυβος"
    stripped = text.strip()
    length = len(stripped)
    
    should_skip = False
    skip_reason = ""

    if length < 5:
        should_skip = True
        skip_reason = "Too short (<5)"
    
    elif length < 15:
        # Smart Filter για μικρές προτάσεις (5-15 chars)
        # Κρατάμε αν έχει αριθμό (π.χ. "Ν. 1234") ή είναι Title Case (π.χ. "Αθήνα")
        import re
        has_digit = bool(re.search(r'\d', stripped))
        is_title = stripped[0].isupper()
        
        # Αν δεν έχει ούτε αριθμό ούτε κεφαλαίο -> Σκουπίδι
        if not has_digit and not is_title:
             should_skip = True
             skip_reason = "Noise/Fragment"
        
        # Αν είναι όλα κεφαλαία -> Πιθανός τίτλος εγγράφου (π.χ. "ΜΕΡΟΣ Α") -> Skip
        if stripped.isupper() and not has_digit:
             should_skip = True
             skip_reason = "All Caps Header"

    if should_skip:
        # IMPORTANT: Mark as 'skipped' or 'approved' (empty) so we don't fetch it again eternally.
        # We will use 'approved' to signify "Process Completed (Result: Trash)"
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("UPDATE sentences SET status='approved' WHERE id=?", (sent_id,))
                # Also delete annotations if any? No, keep history.
        except Exception as e:
            pass
            
        return f"\n⏩ Fast-Skipped ID: {sent_id} | Text: '{stripped}' ({skip_reason}) [Marked Approved]"

    # UX: Print start indicator immediately (so user knows it's not stuck)
    print(f"⏳ Started analyzing Sentence {sent_id}...", flush=True)

    input_log = f"\n🔹 Processing Sentence ID: {sent_id}\n"
    input_log += f"📝 Text: {text[:100]}..." if len(text) > 100 else f"📝 Text: {text}"
    
    current_anns = agent.get_annotations_for_sentence(sent_id)
    input_log += f"\\n📊 Initial Annotations: {len(current_anns)}"
    
    # TIMING START
    t0 = time.time()
    result_log = ""
    
    try:
        # Pass 1: Run Analysis
        result = agent.analyze_and_fix(sent_id, text, current_anns)
        
        # Pass 2: Apply Changes 
        # CAUTION: Enable this ONLY when ready to write to DB
        agent.apply_batch_fixes(sent_id, text, result) 
        
    except Exception as e:
        import traceback
        return input_log + f"\\n❌ Error: {e}\\n" + traceback.format_exc()
        
    duration = time.time() - t0
    result_log += f"\\n⏱️ Time Taken: {duration:.2f}s"
    
    if result:
        corrections = result.get('corrections', [])
        new_entities = result.get('new_entities', [])
        
        result_log += f"\\n   👉 Corrections: {len(corrections)}"
        for c in corrections:
            result_log += f"\\n      - {c['action']} ID {c['id']}: {c.get('text', '')} {c.get('reason', '')}"
            
        result_log += f"\\n   👉 New Entities: {len(new_entities)}"
        for n in new_entities:
            result_log += f"\\n      - FOUND: '{n['text']}' ({n['label']})"
            
    return input_log + result_log

def run_agents_loop():
    agent = AutoAgent()
    
    # ADJUSTED PARALLELISM
    # Reduced from 8 to 4 to prevent Rate Limit (429) deadlocks.
    MAX_WORKERS = 4 
    
    print(f" Starting Autonomous Agents TEST RUN (PARALLEL MODE - {MAX_WORKERS} Threads)...", flush=True)
    
    # TEST LIMIT (Set to 15000 for full run)
    LIMIT = 20000
    OFFSET = 0
    
    sentences = agent.get_pending_sentences_with_annotations(LIMIT, OFFSET)
    
    if not sentences:
        print("No pending sentences found.")
        return
        
    print(f"Processing batch of {len(sentences)} sentences...", flush=True)
    
    total_start = time.time()
    
    # Use ThreadPoolExecutor for Outer Loop
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(process_single_sentence, agent, sent_id, text): sent_id 
            for sent_id, text in sentences
        }
        
        # Wrapped in tqdm for progress tracking
        for future in tqdm(as_completed(future_to_id), total=len(sentences), desc="Sweeping Agent Progress", unit="sent"):
            sent_id = future_to_id[future]
            try:
                log_output = future.result()
                # Use tqdm.write so the bar doesn't break
                tqdm.write(log_output)
            except Exception as e:
                tqdm.write(f"CRITICAL FAULT on Sent {sent_id}: {e}")

    total_duration = time.time() - total_start
    print(f"\\n✅ Test Run Complete. Total Time: {total_duration:.2f}s")

if __name__ == "__main__":
    run_agents_loop()
