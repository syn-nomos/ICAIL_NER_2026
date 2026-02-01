import os
import json
import requests
import time
import random
import difflib
import streamlit as st
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Φόρτωση μεταβλητών περιβάλλοντος από το .env αρχείο
load_dotenv()

class LLMJudge:
    """
    LLM Judge Client for Legal-Hydra (via OpenRouter).
    Uses 'DeepSeek R1' (Reasoning) or 'Claude 3.5 Sonnet' for high-level conflict resolution.
    """
    def __init__(self, model="deepseek/deepseek-r1", api_key=None):
        # Retrieve API Key from various sources
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            try:
                self.api_key = st.secrets["OPENROUTER_API_KEY"]
            except:
                pass

        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Headers specifically for OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://legal-hydra.local", 
            "X-Title": "Legal-Hydra Research"
        }

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """General Helper for Smart R1 API Calls"""
        if not self.api_key:
            return None
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1, # Low temp for rigorous logic
            "max_tokens": 8192, # Ensure long responses aren't truncated
             # Note: R1 might not respect 'json_object' mode perfectly yet, but standard models do.
            # "response_format": {"type": "json_object"} 
        }

        MAX_RETRIES = 5
        BASE_DELAY = 3 # Seconds

        for attempt in range(MAX_RETRIES):
            try:
                # Increased timeout for Reasoning Models (DeepSeek R1/Speciale)
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    # Clean up Thinking Tags from DeepSeek R1 (<think>...</think>)
                    if "<think>" in content:
                        content = content.split("</think>")[-1].strip()
                    
                    # Clean up Markdown JSON blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                        
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback 1: Regex Extraction if Extra Data exists
                        import re
                        match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if match:
                            try:
                                return json.loads(match.group(1))
                            except:
                                pass
                        
                        # Fallback 2: Repair Truncated JSON (Common with long responses)
                        try:
                            # 1. Find the last valid closing brace/bracket
                            last_brace = content.rfind('}')
                            last_bracket = content.rfind(']')
                            cut_off = max(last_brace, last_bracket)
                            
                            if cut_off != -1:
                                truncated = content[:cut_off+1]
                                # 2. Balance braces/brackets
                                num_braces = truncated.count('{') - truncated.count('}')
                                num_brackets = truncated.count('[') - truncated.count(']')
                                
                                # Append needed closers (order matters: usually ] then })
                                # Assumption: We are usually inside a dict or list. 
                                # If nested mixed, this simple count might fail, but worth a try.
                                repaired = truncated + (']' * max(0, num_brackets)) + ('}' * max(0, num_braces))
                                repaired_json = json.loads(repaired)
                                print(f"🔧 Successfully repaired truncated JSON. (Kept {len(str(repaired_json))} chars)")
                                return repaired_json
                        except Exception as e:
                            # print(f"JSON Repair Failed: {e}")
                            pass

                        print(f"LLM JSON Error. Content: {content[:100]}...")
                        # Retry on JSON Error
                        wait_time = BASE_DELAY
                        print(f"⚠️ JSON Parse Error (Attempt {attempt+1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                        
                elif response.status_code == 429: # Rate Limit
                    wait_time = BASE_DELAY * (2 ** attempt) + (random.random() * 2)
                    print(f"⚠️ 429 Rate Limit (Attempt {attempt+1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code >= 500: # Server Error
                    wait_time = BASE_DELAY * (2 ** attempt)
                    print(f"⚠️ Server Error {response.status_code} (Attempt {attempt+1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    print(f"LLMJudge Error {response.status_code}: {response.text}")
                    return None

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait_time = BASE_DELAY * (2 ** attempt)
                print(f"⚠️ Network/Timeout Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            except Exception as e:
                print(f"LLMJudge Exception: {e}")
                return None
        
        print(f"❌ LLMJudge Failed after {MAX_RETRIES} retries.")
        return None

    def resolve_conflict(self, text: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        SOTA Conflict Resolution με DeepSeek V3.
        Κάνει Comparative Analysis μεταξύ των υποψηφίων.
        """
        if not self.api_key:
            return None

        # 1. Δημιουργία λίστας επιλογών για το Prompt
        options_text = ""
        for i, c in enumerate(candidates):
            options_text += f"Option {i+1}: '{c['text']}' -> LABEL: {c['label']} (Source: {c['source']})\n"

        # 2. SOTA System Prompt (Detailed Decision Matrix)
        system_prompt = """
        Είσαι ο Αρχισυντάκτης Νομικής Βάσης Δεδομένων (Senior Legal Editor).
        Καλείσαι να επιλύσεις μια ΣΥΓΚΡΟΥΣΗ (Conflict) μεταξύ δύο υποψήφιων οντοτήτων που αλληλοκαλύπτονται.
        
        CRITERIA MATRIX (Βαθμολόγησε νοητά βάσει αυτών):
        1. **Πληρότητα (Completeness)**: Προτίμησε την οντότητα που περιλαμβάνει ΟΛΟ τον τίτλο.
           - Π.χ. "Υπουργείο Δικαιοσύνης" (ΝΙΚΗΤΗΣ) vs "Δικαιοσύνης" (ΧΑΜΕΝΟΣ).
           - Π.χ. "Ν. 4172/2013" (ΝΙΚΗΤΗΣ) vs "4172/2013" (ΧΑΜΕΝΟΣ).
        
        2. **Καθαρότητα Ορίων (Boundary Hygiene)**: Απέφυγε 'σκουπίδια' στα άκρα.
           - Αν η Option 1 είναι "του Δήμου" (έχει πρόθεση) και η Option 2 είναι "Δήμου", προτίμησε την Option 2.
           - Εξαίρεση: Αν η πρόθεση είναι μέρος του ονόματος (π.χ. "Άγιος Νικόλαος").

        3. **Εξειδίκευση (Specifity)**:
           - "Εφετείο Αθηνών" (ORG/FACILITY) > "Αθηνών" (GPE).

        ΟΔΗΓΙΑ:
        Σύγκρινε τις επιλογές. Αν η μία είναι υποσύνολο της άλλης, διάλεξε την ΠΛΗΡΕΣΤΕΡΗ (Longest Match) αρκεί να είναι γραμματικά σωστή.
        
        Επίστρεψε JSON:
        {
            "analysis": "<σύντομη σύγκριση A vs B>",
            "best_option_index": <ο αριθμός της καλύτερης επιλογής, π.χ. 1 ή 2>
        }
        """

        # 3. Το User Prompt
        user_prompt = f"""
        Κείμενο Πλαισίου: "{text}"
        
        Συγκρουόμενες Επιλογές:
        {options_text}
        
        Ποια είναι η ΒΕΛΤΙΣΤΗ νομική οντότητα;
        """

        return self._call_llm(system_prompt, user_prompt)

    def evaluate_ambiguity(self, text: str, entity_text: str, label1: str, label2: str) -> str:
        """
        SOTA Semantic Parser για διάκριση GPE vs LOCATION.
        Χρησιμοποιεί "Substitution Test" και Semantic Role Labeling.
        """
        if not self.api_key: return None

        system_prompt = """
        Είσαι Συντακτικός και Σημασιολογικός Αναλυτής Νομικών Κειμένων.
        Καλείσαι να επιλύσεις μια ασάφεια μεταξύ GPE (Διοικητική Αρχή) και LOCATION (Γεωγραφικός Τόπος).

        CRITICAL THINKING PROCESS (Εκτέλεσε αυτά τα βήματα):
        1. **Semantic Role Check**: Η οντότητα είναι ΔΡΩΝ ΥΠΟΚΕΙΜΕΝΟ (Agent) ή ΠΑΘΗΤΙΚΟΣ ΧΩΡΟΣ (Location);
           - Agent (αποφασίζει, δικάζει, ενάγει, πληρώνει) -> GPE.
           - Container (βρίσκεται, κείται, εντός, επί της οδού) -> LOCATION.
        
        2. **The Substitution Test (Το Κλειδί)**:
           - Προσπάθησε να αντικαταστήσεις την οντότητα με τη λέξη "ΔΙΟΙΚΗΣΗ/ΔΗΜΟΣΙΟ". Αν βγάζει νόημα -> GPE.
           - Προσπάθησε να αντικαταστήσεις την οντότητα με τη λέξη "ΠΕΡΙΟΧΗ/ΤΟΠΟΣ". Αν βγάζει νόημα -> LOCATION.

        3. **Legal Indicator Check**:
           - "Κατά του [X]" (Δικαστική προσφυγή) -> GPE.
           - "Στο [X]", "κάτοικος [X]" -> LOCATION.

        Επίστρεψε JSON:
        {
            "substitution_test": "<ποια λέξη ταίριαξε καλύτερα: 'Διοίκηση' ή 'Περιοχή'>",
            "reasoning": "<σύντομη αιτιολόγηση στα Ελληνικά>",
            "selected_label": "GPE" ή "LOCATION",
            "confidence": <0.0 - 1.0>
        }
        """

        user_prompt = f"""
        Κείμενο Πλαισίου: "{text}"
        Οντότητα: "{entity_text}"
        
        Επίλυσε την ασάφεια {label1} vs {label2}.
        """
        
        result = self._call_llm(system_prompt, user_prompt)
        # Αυξάνουμε το threshold στο 0.7 για να είμαστε σίγουροι
        if result and result.get('confidence', 0) > 0.7:
            print(f"🧠 Ambiguity Logic: {result.get('substitution_test')} | {result.get('reasoning')}")
            return result.get('selected_label')
        return None

    def _find_best_match_span(self, context: str, candidate: str, threshold: float = 0.80) -> Optional[str]:
        """
        Εντοπίζει το τμήμα του 'context' που ταιριάζει καλύτερα στο 'candidate'.
        Χρησιμοποιείται για να διορθώσει μικρο-αλλαγές που έκανε το LLM (π.χ. πτώσεις, ορθογραφία)
        και να επιστρέψει ΤΟ ΑΚΡΙΒΕΣ ΚΕΙΜΕΝΟ από το context.
        """
        if not candidate or not context: return None
        if candidate in context: return candidate # Exact match optimization
        
        normalized_candidate = candidate.lower().strip()
        best_ratio = 0.0
        best_span = None
        
        target_len = len(candidate)
        # Ψάχνουμε σε παράθυρο μήκους +/- 20% του candidate
        min_len = max(1, int(target_len * 0.8))
        max_len = min(len(context), int(target_len * 1.2))
        
        matcher = difflib.SequenceMatcher(None, normalized_candidate, "")
        
        # Sliding window search (Optimized for short contexts)
        for length in range(min_len, max_len + 1):
             for i in range(len(context) - length + 1):
                sub = context[i : i + length]
                # Compare normalized
                matcher.set_seq2(sub.lower())
                ratio = matcher.ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = sub
        
        if best_ratio >= threshold:
            return best_span
        return None

    def refine_boundaries(self, text_context: str, entity_text: str, label: str, user_feedback: bool = False) -> str:
        """
        Smart Refinement: Τρία Εξειδικευμένα Prompts ανάλογα με την κατηγορία.
        """
        if not self.api_key: return entity_text


        # --- CASE 1: LEG-REFS (ΝΟΜΟΙ, ΚΩΔΙΚΕΣ, Π.Δ., ΟΔΗΓΙΕΣ) ---
        if label in ['LEG-REFS', 'LEG_REFS']:
            system_prompt = """
            Είσαι διορθωτής ορίων νομικών οντοτήτων (Boundary Refiner).
            Στόχος: Εντόπισε τα ΑΚΡΙΒΗ όρια της νομικής αναφοράς (Τύπος + Αριθμός + Τίτλος).

            CRITICAL RULES:
            1. **LONG DISTANCE EXPANSION**:
               - Η πλήρης αναφορά συχνά περιέχει παρεμβαλλόμενο κείμενο.
               - Μοτίβο: `[Νόμος/ΠΔ] [Αριθμός] ... (ΦΕΚ) ... «Τίτλος»`.
               - ΠΡΕΠΕΙ να συμπεριλάβεις ΟΛΟΚΛΗΡΟ το απόσπασμα μέχρι το κλείσιμο του τίτλου, αρκεί να αναφέρεται στον ίδιο νόμο.
               - Context: "...του ν. 4000/2012, όπως τροποποιήθηκε και δημοσιεύθηκε στο (Α' 88), φέροντας τον τίτλο «Περί Ευθύνης»..."
               - Output: "ν. 4000/2012, όπως τροποποιήθηκε και δημοσιεύθηκε στο (Α' 88), φέροντας τον τίτλο «Περί Ευθύνης»" (Αν αυτό είναι το "span" της αναφοράς).
               - ΠΡΟΣΟΧΗ: Αν το ενδιάμεσο κείμενο είναι πολύ μεγάλο (> 15 λέξεις) και άσχετο, τότε ΣΤΑΜΑΤΑ στον αριθμό/ΦΕΚ.

            2. **TITLE CAPTURE**:
               - Αν ακολουθεί τίτλος σε εισαγωγικά, είναι σχεδόν πάντα μέρος της οντότητας.

            3. **CLEANING (Start Only)**:
               - Κόψε ΜΟΝΟ τα αρχικά "σκουπίδια" (άρθρα/προθέσεις).
               - Input: "του ν. 4000" -> Output: "ν. 4000"

            4. **CONSERVATISM (ΑΣΦΑΛΕΙΑ)**:
               - Αν τα όρια είναι ήδη σωστά (περιέχουν Νόμο + Αριθμό + Τίτλο αν υπάρχει), ΕΠΙΣΤΡΕΨΕ ΤΟ ΚΕΙΜΕΝΟ ΑΚΡΙΒΩΣ ΟΠΩΣ ΕΙΝΑΙ.
               - Μην αλλάζεις κάτι αν δεν είσαι 100% σίγουρος ότι βελτιώνεται.

            Επίστρεψε JSON: { "refined_text": "<το απόσπασμα>" }
            """

        # --- CASE 2: PUBLIC-DOCS (ΑΠΟΦΑΣΕΙΣ, ΕΓΚΥΚΛΙΟΙ, Υ.Α.) ---
        elif label in ['PUBLIC-DOCS', 'DOCS']:
            system_prompt = """
            Είσαι διορθωτής ορίων δημοσίων εγγράφων (Public Docs Refiner).
            Στόχος: Εντόπισε τα ΑΚΡΙΒΗ όρια της αναφοράς (Αριθμός + Τίτλος).

             CRITICAL RULES:
            1. **PATTERN RECOGNITION**:
               - Αναζήτησε το μοτίβο: `[Τύπος Εγγράφου] [Αριθμός/Πρωτόκολλο] ... [Τίτλος/Θέμα]`.
               - Παράδειγμα: "Υ.Α. 12345/2020 (Β' 500) «Περί καθορισμού όρων...»".
            
            2. **GAP JUMPING (The Semantic Link)**:
               - Ο τίτλος μπορεί να βρίσκεται αρκετές λέξεις μετά τον αριθμό. 
               - ΑΝ το ενδιάμεσο κείμενο είναι τυπικό (π.χ. "του Υπουργείου..." ή "με θέμα") ΣΥΜΠΕΡΙΛΑΒΕ ΤΟΝ ΤΙΤΛΟ.
               - ΑΝ το ενδιάμεσο κείμενο αλλάζει θέμα, ΣΤΑΜΑΤΑ στον αριθμό.
               - Context: "η υπ' αριθμ. 500 απόφαση του Υπουργού που αφορά «Μέτρα στήριξης»" -> Output: "αριθμ. 500 απόφαση του Υπουργού ... «Μέτρα στήριξης»" (ΠΡΟΣΟΧΗ: Επίστρεψε το συνεχόμενο string από το κείμενο, μην κόβεις ενδιάμεσα).
               
            3. **CLEANING**:
               - Κόψε προθέσεις στην αρχή ("δυνάμει της", "με την").
               - Input: "με την Υ.Α. 123" -> Output: "Υ.Α. 123".

            4. **LONG ENTITIES**:
               - Μην φοβηθείς να επιστρέψεις μεγάλο string αν ο τίτλος είναι μακρύς.

            5. **CONSERVATISM (ΑΣΦΑΛΕΙΑ)**:
               - Αν η είσοδος είναι ήδη σωστή και περιλαμβάνει πλήρως την αναφορά, ΜΗΝ την αλλάξεις.

            Επίστρεψε JSON: { "refined_text": "<το ακριβές απόσπασμα>" }
            """

        # --- CASE 3: GENERAL ENTITIES (ORG, GPE, FACILITY, PERSON, LOCATION) ---
        else:
            system_prompt = """
            Είσαι Γλωσσολόγος διορθωτής οντοτήτων (NLP Refiner).
            Καθαρίζεις τα "σκουπίδια" από τα άκρα των οντοτήτων και εξασφαλίζεις ότι οι λέξεις είναι ολόκληρες.

            RULES:
            1. **TRIM ARTICLES & PREPOSITIONS**: 
               - Βγάλε τα "ο, η, το, του, την, τον, οι, των" και "σε, από, με, για".
               - ΕΞΑΙΡΕΣΗ (PERSON PATRONYMICS): Αν η οντότητα είναι PERSON και περιέχει "του [ΟΝΟΜΑ]" που δηλώνει πατρώνυμο, ΜΗΝ το αφαιρέσεις αν είναι στη μέση ή στο τέλος. 
               - Π.χ. "Γεώργιος Παπαδόπουλος του Ιωάννη" -> ΚΡΑΤΑ το "του Ιωάννη".
               - "του Γεωργίου Παπαδόπουλου" -> "Γεωργίου Παπαδόπουλου" (Εδώ το 'του' είναι στην αρχή, άρα φεύγει).

            2. **HEALING BROKEN WORDS (CRITICAL)**:
               - Αν η οντότητα φαίνεται να ξεκινάει στη μέση λέξης (π.χ. "ΝΗΣ ΚΩΝΣΤΑΝΤΙΝΙΔΗΣ" ή "στική Αρχή"), ΚΟΙΤΑ ΤΟ CONTEXT αριστερά.
               - Αν η λέξη στο text είναι "ΙΩΑΝΝΗΣ", και το entity είναι "ΝΗΣ", ΣΥΜΠΛΗΡΩΣΕ ΤΟ.
               - Output: "ΙΩΑΝΝΗΣ ΚΩΝΣΤΑΝΤΙΝΙΔΗΣ".

            3. **FACILITY/ORG PROTECTION**:
               - ΜΗΝ αφαιρείς ουσιαστικά που δηλώνουν τον τύπο της οντότητας.
               - ΛΑΘΟΣ: "Λιμάνι της Θεσσαλονίκης" -> "Θεσσαλονίκη" (Αυτό είναι Location, όχι Facility).
               - ΣΩΣΤΟ: "Λιμάνι της Θεσσαλονίκης" -> "Λιμάνι της Θεσσαλονίκης" (Εδώ το 'Λιμάνι' είναι απαραίτητο, ίσως βγάλεις μόνο το 'στο' αν υπήρχε πριν).
               - Αν το input είναι "στο Λιμάνι της Θεσσαλονίκης", Output -> "Λιμάνι της Θεσσαλονίκης".

            4. **CONSERVATISM (ΑΣΦΑΛΕΙΑ)**:
               - Αν η οντότητα είναι ήδη καθαρή και σωστή, ΜΗΝ την αλλάξεις. Μην αφαιρείς πληροφορία που είναι μέρος του ονόματος.

            Επίστρεψε JSON: { "refined_text": "..." }
            """

        instructions = ""
        if user_feedback:
            instructions = """
        CONTEXT: Ο χρήστης έχει επιβεβαιώσει ότι η οντότητα είναι VALID (υπαρκτή) αλλά έχει ΛΑΘΟΣ ΟΡΙΑ (Boundaries).
        Στόχος: Κράτα τον σημαντικό πυρήνα της οντότητας και φτιάξε τα άκρα (κόψε σκουπίδια ή πρόσθεσε ότι λείπει).
        """

        user_prompt = f"""
        Κείμενο Πλαισίου: "{text_context}"
        Αρχική Οντότητα: "{entity_text}"
        Ετικέτα: {label}
        {instructions}
        Διόρθωσε την οντότητα σύμφωνα με τους κανόνες της κατηγορίας {label}.
        """

        response = self._call_llm(system_prompt, user_prompt)
        if response and response.get('refined_text'):
            refined_candidate = response['refined_text']
            
            # --- SAFETY CHECK: FUZZY SPAN PROJECTION ---
            # Αντί να απορρίψουμε τελείως αν δεν υπάρχει (exact match),
            # προσπαθούμε να βρούμε το αντίστοιχο span στο κείμενο.
            # Αν το LLM άλλαξε την πτώση ("Υπουργείο" αντί "Υπουργείου"), 
            # εμείς θα πάρουμε το "Υπουργείου" από το κείμενο.
            
            matched_span = self._find_best_match_span(text_context, refined_candidate, threshold=0.85)
            
            if matched_span:
                return matched_span
            else:
                # Αν δεν υπάρχει ούτε fuzzy match, τότε το LLM hallucinated heavily.
                print(f"⚠️ Safety Rejection: Refined text '{refined_candidate}' too different from context. Reverting.")
                return entity_text

        return entity_text

    def find_missing_entities(self, text: str, existing_entities: List[str]) -> List[Dict[str, Any]]:
        """
        Ψάχνει για οντότητες που ΧΑΣΑΜΕ τελείως.
        Χρησιμοποίησε το για Auditing.
        """
        if not self.api_key: return []

        system_prompt = """
        Είσαι "κυνηγός λαθών" σε σύστημα NER για Ελληνικά Νομικά Κείμενα.
        Σου δίνεται ένα κείμενο και μια λίστα με Οντότητες που ΗΔΗ βρήκαμε.
        
        Ο Στόχος σου: Βρες τι ΞΕΧΑΣΑΜΕ.
        
        Ψάξε για:
        - ORG (Οργανισμούς, Υπουργεία)
        - LEG-REFS (Νόμους, Άρθρα)
        - GPE / FACILITY / LOCATION
        
        ΜΗΝ επιστρέψεις οντότητες που ήδη υπάρχουν στη λίστα "Existing".
        ΜΗΝ επιστρέψεις γενικές λέξεις (π.χ. "ο νόμος", "η απόφαση") χωρίς αριθμό/προσδιορισμό.
        
        Επίστρεψε JSON:
        {
            "missing_entities": [
                {"text": "...", "label": "...", "reason": "..."}
            ]
        }
        """
        
        existing_str = ", ".join([f"'{e}'" for e in existing_entities])
        user_prompt = f"""
        Κείμενο: "{text}"
        Ήδη Γνωστές Οντότητες (IGNORE THESE): [{existing_str}]
        
        Υπάρχει κάτι άλλο που είναι νομική οντότητα και το χάσαμε;
        """
        
        response = self._call_llm(system_prompt, user_prompt)
        if response and 'missing_entities' in response:
            safe_entities = []
            for item in response['missing_entities']:
                 candidate_text = item.get('text', '')
                 # Verify existence with Fuzzy Match - FORCE mapping to existing text
                 matched_span = self._find_best_match_span(text, candidate_text, threshold=0.70)
                 if matched_span:
                     item['text'] = matched_span # Use matched text from SOURCE text
                     safe_entities.append(item)
                 else:
                     # Even if not found via fuzzy match, return the original candidate 
                     # The frontend is now robust enough to find it (using S5 strategy)
                     # Or we can mark it as unverified
                     print(f"⚠️ Missing Entity Exact Match Failed: '{candidate_text}' (Keeping original)")
                     safe_entities.append(item)
            return safe_entities
        return []

    def validate_entity(self, text: str, entity_text: str, label: str) -> Dict[str, bool]:
        """
        Εκτελεί 'Συμβούλιο' (Council) 3 μοντέλων για την αξιολόγηση της οντότητας.
        Στόχος: Μέγιστη ακρίβεια στις 'Gray Zone' περιπτώσεις.
        """
        if not self.api_key: return {'is_valid': True, 'boundary_error': False}

        # Ορίζουμε το Συμβούλιο (Council of Judges)
        # ΣΗΜΕΙΩΣΗ: Τα τοπικά μοντέλα (RoBERTa) δεν μπορούν να κάνουν Validation Reasoning (είναι μόνο για εντοπισμό).
        # Επιλέγουμε τον βέλτιστο συνδυασμό "Cheap & Good" (High IQ + Low Cost):
        council_models = [
            self.model,                          # 1. Primary Judge (DeepSeek V3 - SOTA & Cheap)
            "google/gemini-2.0-flash-exp:free",  # 2. The Linguist (Gemini 2.0 - Free Tier)
            "meta-llama/llama-3.3-70b-instruct"  # 3. The Skeptic (Llama 3.3 - Best Value High-Intelligence)
        ]

        system_prompt = """
        Είσαι AΥΣΤΗΡΟΤΑΤΟΣ κριτής νομικών οντοτήτων (Entity Validator).
        Αποστολή: Αξιολόγησε αν η υποψήφια οντότητα είναι ΣΩΣΤΗ, ΛΑΘΟΣ ή ΕΛΛΙΠΗΣ, βάσει των επίσημων ορισμών.

        ΟΡΙΣΜΟΙ ΚΑΤΗΓΟΡΙΩΝ:
        - **ORG**: Νομικά πρόσωπα, Εταιρείες, Υπουργεία, Δικαστήρια. ΟΧΙ ρόλοι ("Πρόεδρος") ούτε Φυσικά Πρόσωπα.
        - **GPE**: Διοικητικές Αρχές (Κράτη, Δήμοι). ΟΧΙ απλές τοποθεσίες.
        - **PERSON**: Ονοματεπώνυμα. ΟΧΙ τίτλοι μόνοι τους.
        - **LEG-REFS**: Νόμοι (ν.), Π.Δ., Κώδικες.
        - **PUBLIC-DOCS**: ΦΕΚ, Αποφάσεις, Εγκύκλιοι (με αριθμό).
        - **FACILITY**: Κτίρια, Υποδομές, Οδοί.

        CRITERIA MATRIX:
        1. **INVALID (is_valid: False)**:
           - Ρόλοι χωρίς όνομα (π.χ. "ο Υπουργός", "ο Διοικητής"). -> ΛΑΘΟΣ.
           - Γενικές λέξεις (π.χ. "σύμφωνα με", "το άρθρο"). -> ΛΑΘΟΣ.
           - Σφάλμα Κατηγορίας (π.χ. "Αθήνα" ως ORG). -> ΛΑΘΟΣ.

        2. **BOUNDARY ERROR (boundary_error: True)**:
           - **INCOMPLETE**: Κόπηκε μέρος του ονόματος (π.χ. "Υπουργείο" αντί "Υπουργείο Υγείας").
           - **DIRTY**: Έχει σκουπίδια (π.χ. "του Γιώργου" -> το "του" είναι λάθος).

        3. **VALID (is_valid: True)**:
           - Η οντότητα είναι πλήρης, σωστή και ανήκει στην κατηγορία της.

        Επίστρεψε JSON: { "is_valid": boolean, "boundary_error": boolean }
        """
        
        user_prompt = f"""
        Κείμενο: "{text}"
        Υποψήφια Οντότητα: "{entity_text}"
        Ετικέτα: {label}
        """

        # Συνάρτηση για κλήση ενός μοντέλου
        def ask_judge(model_id):
            try:
                # Προσαρμογή payload για κάθε μοντέλο αν χρειάζεται (εδώ το πρότυπο OpenAI είναι κοινό)
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
                # Σημείωση: Το Llama 3.3 στο OpenRouter ίσως θέλει "json_mode" ή απλά prompt instruction
                # Το DeepSeek και το Gemini υποστηρίζουν JSON mode καλά.
                
                resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content']
                    # Καθαρισμός JSON markdown
                    if "```json" in content:
                        content = content.replace("```json", "").replace("```", "")
                    return json.loads(content)
            except:
                pass
            return None

        # Παράλληλη εκτέλεση (Multithreading)
        votes = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_model = {executor.submit(ask_judge, m): m for m in council_models}
            for future in as_completed(future_to_model):
                res = future.result()
                if res and isinstance(res, dict):
                    votes.append(res)
        
        if not votes:
            # Fallback if all fail
            return {'is_valid': True, 'boundary_error': False} # Assume innocent until proven guilty? Or opposite? Let's say valid to check boundaries manually later.

        # --- VOTING LOGIC ---
        # 1. Count Votes
        valid_votes = sum(1 for v in votes if v.get('is_valid') is True)
        invalid_votes = sum(1 for v in votes if v.get('is_valid') is False)
        boundary_votes = sum(1 for v in votes if v.get('boundary_error') is True)
        
        total_votes = len(votes)
        
        print(f"🗳️ Council Votes ({total_votes}): Valid={valid_votes}, Invalid={invalid_votes}, Boundary={boundary_votes}")

        # Decision 1: Is it completely Invalid? (Majority Rule)
        if invalid_votes > valid_votes and boundary_votes < (total_votes / 2):
            # Αν η πλειοψηφία λέει INVALID και δεν υπάρχει ισχυρή ένδειξη Boundary Error -> TRASH IT.
            return {'is_valid': False, 'boundary_error': False}
        
        # Decision 2: Boundary Error Priority
        # Αν έστω και το 1/3 των μοντέλων (δηλ. 1 στα 3) εντοπίσει Boundary Error, δίνουμε ΣΗΜΑ ΓΙΑ ΔΙΟΡΘΩΣΗ.
        # Είμαστε "Risk Averse" στο να χάσουμε μέρος της οντότητας.
        if boundary_votes >= 1: 
            return {'is_valid': False, 'boundary_error': True}
        
        # Decision 3: Default Valid
        return {'is_valid': True, 'boundary_error': False}

    # [REMOVED DUPLICATE _call_llm METHOD]


    def verify_entity(self, text: str, label: str, context: str) -> bool:
        """
        Επαληθεύει αν μια οντότητα είναι σωστή μέσα στο context.
        """
        if not self.api_key: return False

        prompt = f"""
        Context: "{context}"
        Entity: "{text}"
        Proposed Label: {label}
        
        Is this annotation correct for a Greek Legal Text?
        Reply ONLY with 'YES' or 'NO'.
        """
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a strict legal annotator. Reply YES or NO."},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=5
            )
            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content'].strip().upper()
                return "YES" in answer
        except:
            pass
        return False

    def ask_question(self, sentence: str, question: str, segment: str = None) -> str:
        """
        Ρωτάει τον LLM μια ερώτηση σχετικά με την πρόταση ή ένα τμήμα της.
        """
        if not self.api_key:
            return "Error: OPENROUTER_API_KEY not found."

        entity_definitions = """
        ΟΡΙΣΜΟΙ ΟΝΤΟΤΗΤΩΝ (Ελληνικό Νομικό Πλαίσιο):
        1. PERSON (Πρόσωπο): Φυσικά πρόσωπα (Ονοματεπώνυμα). Εξαιρούνται γενικοί ρόλοι (π.χ. "ο Δικηγόρος") εκτός αν συνοδεύονται από όνομα.
        2. ORG (Οργανισμός): Εταιρείες, Τράπεζες, Ιδρύματα, Υπουργεία, Πολιτικά Κόμματα, Σύλλογοι. (π.χ. "Alpha Bank", "Υπουργείο Οικονομικών").
        3. GPE (Γεωπολιτική Οντότητα): Χώρες, Πόλεις, Δήμοι, Περιφέρειες. Συχνά όταν αναφέρονται ως ΔΙΟΙΚΗΣΗ ή ΑΡΧΗ. (π.χ. "Ο Δήμος Αθηναίων", "Το Ελληνικό Δημόσιο").
        4. FACILITY (Εγκατάσταση): Κτίρια, Αεροδρόμια, Λιμάνια, Οδοί, Πλατείες, Νοσοκομεία (ως κτίρια). (π.χ. "Αεροδρόμιο Ελ. Βενιζέλος", "Οδός Πανεπιστημίου").
        5. LEG_REFS (Νομική Αναφορά): Νόμοι, Προεδρικά Διατάγματα, Κώδικες, Άρθρα. (π.χ. "Ν. 4172/2013", "άρθρο 5 παρ. 1", "Αστικός Κώδικας"). Αν υπάρχει τίτλος δίπλα στην αναφορά, μέσα σε εισαγωγικά, περιέλαβέ τον.
        6. PUBLIC_DOCS (Δημόσιο Έγγραφο): ΦΕΚ, Εγκύκλιοι, Αριθμοί Αποφάσεων, Συμβάσεις. (π.χ. "ΦΕΚ Α' 120/2010", "Πολ. 1020/2014").  Αν υπάρχει τίτλος δίπλα στην αναφορά, μέσα σε εισαγωγικά, περιέλαβέ τον.
        7. DATE (Ημερομηνία): Ημερομηνίες, Έτη, Μήνες, Χρονικές Περίοδοι. (π.χ. "1η Ιανουαρίου 2024", "το έτος 2020").
        8. LOCATION (Τοποθεσία): Φυσικά γεωγραφικά στοιχεία (βουνά, λίμνες, ποτάμια, νησιά) ΚΑΙ γεωγραφικοί προσδιορισμοί χωρίς διοικητική χροιά. (π.χ. "Ποταμός Έβρος", "Νήσος Κρήτη", "Αθήνα" ως τόπος που συνέβει κατι).
        
        ΔΙΑΚΡΙΣΗ GPE vs LOCATION vs FACILITY:
        - GPE: Έχει διοικητική/πολιτική υπόσταση. Χρησιμοποιείται όταν η οντότητα δρα, αποφασίζει, ή έχει νομική προσωπικότητα (π.χ. "Η Ελλάδα υπέγραψε", "Ο Δήμος αποφάσισε").
        - LOCATION: Απλός γεωγραφικός προσδιορισμός ή φυσικό στοιχείο. Χρησιμοποιείται ως τόπος που συνέβει κατι, κατοικίας, ή φυσικά όρια (π.χ. "γεννήθηκε στην Αθήνα", "ταξίδεψε στην Κρήτη", "στον Όλυμπο").
        - FACILITY: Ανθρωπογενής κατασκευή/κτίριο (π.χ. "στο Μέγαρο Μαξίμου", "στο Αεροδρόμιο").
        """

        system_prompt = f"""
        Είσαι ένας εξειδικευμένος Βοηθός Νομικής Σχολιασμού (Legal NER) για το Ελληνικό Δίκαιο.
        Διαθέτεις κριτική σκέψη και βαθιά κατανόηση της ελληνικής γραμματικής και νομικής ορολογίας.
        
        Στόχος σου είναι να βοηθήσεις τον χρήστη να σχολιάσει σωστά οντότητες, εντοπίζοντας παγίδες που δεν είναι προφανείς.

        {entity_definitions}
        
        ΟΔΗΓΙΕΣ ΑΝΑΛΥΣΗΣ (CRITICAL THINKING):
        1. **Έλεγχος Ορίων (Boundaries):** Πρόσεξε αν η επιλογή περιλαμβάνει κατά λάθος άρθρα (ο, η, το), προθέσεις (σε, από) ή σημεία στίξης. Αυτά ΠΡΕΠΕΙ να εξαιρούνται (εκτός αν είναι μέρος του επίσημου ονόματος π.χ. "Τράπεζα της Ελλάδος").
        2. **Επίλυση Συγκρούσεων (Conflicts):** Αν μια οντότητα μοιάζει με πολλές κατηγορίες (π.χ. "Ελλάδα" -> GPE ή LOCATION;), ανάλυσε το ρήμα και το πλαίσιο. Δρα ως κράτος/διοίκηση (GPE) ή ως τόπος (LOCATION);
        3. **Πληρότητα:** Ελέγξε αν η οντότητα κόπηκε στη μέση (π.χ. "Υπουργείο" αντί "Υπουργείο Οικονομικών").
        
        Απάντησε στην ερώτηση του χρήστη στα ΕΛΛΗΝΙΚΑ.
        Δώσε μια σαφή απάντηση (ΝΑΙ/ΟΧΙ/ΚΑΤΗΓΟΡΙΑ).
        Συμπερίλαβε μια σύντομη αλλά επεξηγηματική αιτιολόγηση και, αν υπάρχει λάθος, ΠΡΟΤΕΙΝΕ τη σωστή διόρθωση/οντότητα.
        """

        user_content = f"Sentence: \"{sentence}\"\n"
        if segment:
            user_content += f"Focus Segment: \"{segment}\"\n"
        
        user_content += f"\nQuestion: {question}"

        # List of models to try in order of preference
        models_to_try = [
            "google/gemini-2.5-flash",          # User requested specific model
            "google/gemini-2.0-flash-exp:free", # Free tier experimental (often works)
            "google/gemini-2.0-flash-exp",      # Standard experimental
            "google/gemini-flash-1.5",          # Stable fallback
        ]

        last_error = ""

        for model in models_to_try:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "model": model, 
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ]
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    last_error = f"Error {response.status_code} on {model}: {response.text}"
                    # If 404 or 400, try next model
                    if response.status_code in [400, 404, 429]:
                        continue
                    else:
                        # Other errors might be transient or auth related, but let's try next anyway
                        continue
            except Exception as e:
                last_error = f"Exception on {model}: {str(e)}"
                continue
        
        return f"All models failed. Last error: {last_error}"

    def stream_question(self, sentence: str, question: str, segment: str = None):
        """
        Streams the LLM response for a question.
        Yields tokens as they arrive.
        """
        if not self.api_key:
            yield "Error: OPENROUTER_API_KEY not found."
            return

        # Reuse the same prompts as ask_question
        entity_definitions = """
        ΟΡΙΣΜΟΙ ΟΝΤΟΤΗΤΩΝ (Ελληνικό Νομικό Πλαίσιο):
        1. PERSON (Πρόσωπο): Φυσικά πρόσωπα (Ονοματεπώνυμα). Εξαιρούνται γενικοί ρόλοι (π.χ. "ο Δικηγόρος") εκτός αν συνοδεύονται από όνομα.
        2. ORG (Οργανισμός): Εταιρείες, Τράπεζες, Ιδρύματα, Υπουργεία, Πολιτικά Κόμματα, Σύλλογοι. (π.χ. "Alpha Bank", "Υπουργείο Οικονομικών").
        3. GPE (Γεωπολιτική Οντότητα): Χώρες, Πόλεις, Δήμοι, Περιφέρειες. Συχνά όταν αναφέρονται ως ΔΙΟΙΚΗΣΗ ή ΑΡΧΗ. (π.χ. "Ο Δήμος Αθηναίων", "Το Ελληνικό Δημόσιο").
        4. FACILITY (Εγκατάσταση): Κτίρια, Αεροδρόμια, Λιμάνια, Οδοί, Πλατείες, Νοσοκομεία (ως κτίρια). (π.χ. "Αεροδρόμιο Ελ. Βενιζέλος", "Οδός Πανεπιστημίου").
        5. LEG_REFS (Νομική Αναφορά): Νόμοι, Προεδρικά Διατάγματα, Κώδικες, Άρθρα. (π.χ. "Ν. 4172/2013", "άρθρο 5 παρ. 1", "Αστικός Κώδικας"). Αν υπάρχει τίτλος δίπλα στην αναφορά, μέσα σε εισαγωγικά, περιέλαβέ τον.
        6. PUBLIC_DOCS (Δημόσιο Έγγραφο): ΦΕΚ, Εγκύκλιοι, Αριθμοί Αποφάσεων, Συμβάσεις. (π.χ. "ΦΕΚ Α' 120/2010", "Πολ. 1020/2014").  Αν υπάρχει τίτλος δίπλα στην αναφορά, μέσα σε εισαγωγικά, περιέλαβέ τον.
        7. DATE (Ημερομηνία): Ημερομηνίες, Έτη, Μήνες, Χρονικές Περίοδοι. (π.χ. "1η Ιανουαρίου 2024", "το έτος 2020").
        8. LOCATION (Τοποθεσία): Φυσικά γεωγραφικά στοιχεία (βουνά, λίμνες, ποτάμια, νησιά) ΚΑΙ γεωγραφικοί προσδιορισμοί χωρίς διοικητική χροιά. (π.χ. "Ποταμός Έβρος", "Νήσος Κρήτη", "Αθήνα" ως τόπος που συνέβει κατι).
        """

        system_prompt = f"""
        Είσαι ένας εξειδικευμένος Βοηθός Νομικής Σχολιασμού (Legal NER) για το Ελληνικό Δίκαιο.
        Διαθέτεις κριτική σκέψη και βαθιά κατανόηση της ελληνικής γραμματικής και νομικής ορολογίας.
        {entity_definitions}
        Απάντησε στην ερώτηση του χρήστη στα ΕΛΛΗΝΙΚΑ.
        """

        user_content = f"Sentence: \"{sentence}\"\n"
        if segment:
            user_content += f"Focus Segment: \"{segment}\"\n"
        user_content += f"\nQuestion: {question}"

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model, 
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_content = line_str[6:]
                            if data_content.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_content)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except:
                                continue
            else:
                yield f"Error {response.status_code}: {response.text}"
        except Exception as e:
            yield f"Exception: {str(e)}"

class CloudScanner:
    """
    Tier 1 Scanner: Meta Llama 3.3 70B Instruct (via OpenRouter).
    High performance, low cost ($0.10/M), excellent Greek support.
    """
    def __init__(self, api_key=None):
        self.api_key = os.getenv("OPENROUTER_API_KEY") or api_key
        self.model_name = "meta-llama/llama-3.3-70b-instruct" 
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.is_ready = True if self.api_key else False

    def scan_text(self, text: str) -> List[Dict[str, Any]]:
        """
        High-Recall Scan using Llama 3.3.
        """
        if not self.is_ready:
            return []

        # Llama 3.3 Prompt (Greek)
        prompt = f"""
        Εξήγαγε τις νομικές οντότητες από το παρακάτω Ελληνικό κείμενο.
        Επίστρεψε ΑΥΣΤΗΡΑ JSON.

        ΟΡΙΣΜΟΙ ΟΝΤΟΤΗΤΩΝ (Reference Guide):
        1. **ORG (Οργανισμοί)**: Νομικά πρόσωπα δημοσίου και ιδιωτικού δικαίου.
           - ΝΑΙ: Υπουργεία ("Υπουργείο Δικαιοσύνης"), Δικαστήρια ("Άρειος Πάγος", "ΣτΕ"), Εταιρείες ("ΟΤΕ", "Alpha Bank"), Ιδρύματα, Σύλλογοι, Επιτροπές.
           - ΟΧΙ: Φυσικά πρόσωπα, Ρόλοι ("Ο Πρόεδρος"), Κτίρια ("Μέγαρο Μαξίμου").
        
        2. **GPE (Γεωπολιτικές Οντότητες)**: Χώρες, Πόλεις, Δήμοι με διοικητική υπόσταση.
           - ΝΑΙ: "Ελλάδα" (ως κράτος), "Δήμος Αθηναίων" (ως αρχή), "Ευρωπαϊκή Ένωση".
           - ΟΧΙ: Απλές τοποθεσίες ("στην Αθήνα"), Φυσικά τοπία.

        3. **PERSON (Φυσικά Πρόσωπα)**: Συγκεκριμένα ονοματεπώνυμα.
           - ΝΑΙ: "Γεώργιος Παπανδρέου", "Κ. Καραμανλής".
           - ΟΧΙ: Τίτλοι ("Ο Πρωθυπουργός", "Ο Υπουργός"), Γενικές αναφορές.

        4. **FACILITY (Εγκαταστάσεις)**: Κτίρια και υποδομές.
           - ΝΑΙ: "Αεροδρόμιο Ελ. Βενιζέλος", "Δικαστικό Μέγαρο", "Φυλακές Κορυδαλλού", "Οδός Πανεπιστημίου".
        
        5. **LEG-REFS (Νομικές Αναφορές)**: Νόμοι και νομοθετήματα.
           - ΝΑΙ: "ν. 4000/2012", "άρθρο 5 του Συντάγματος", "π.δ. 80/2020", "Αστικός Κώδικας".
        
        6. **PUBLIC-DOCS (Δημόσια Έγγραφα)**: Επίσημα έγγραφα και αριθμοί.
           - ΝΑΙ: "ΦΕΚ Α 100", "υπ' αριθμ. 1234 Απόφαση", "Εγκύκλιοι", "ΑΔΑ".
           - ΚΑΝΟΝΑΣ: Περιλαμβάνει τον αριθμό και, αν υπάρχει, τον τίτλο σε εισαγωγικά.

        7. **DATE**: Ημερομηνίες και έτη.

        ΣΗΜΑΝΤΙΚΟΙ ΑΡΝΗΤΙΚΟΙ ΚΑΝΟΝΕΣ:
        1. **ΑΓΝΟΗΣΕ ΡΟΛΟΥΣ**: "Πρόεδρος", "Υπουργός", "Διοικητής", "Βουλευτής" ΔΕΝ είναι οντότητες αν δεν έχουν όνομα. ΜΗΝ τα βγάζεις.
        2. **ΑΓΝΟΗΣΕ ΠΑΡΕΝΘΕΣΕΙΣ**: Μη βγάζεις οντότητες (ειδικά ΦΕΚ) αν βρίσκονται αυστηρά μέσα σε παρένθεση π.χ. "(ΦΕΚ Α 144)".
        3. **ΔΙΑΧΩΡΙΣΕ GPE/LOCATION**: Αν φαίνεται σαν τόπος γέννησης ή διεύθυνση, είναι LOCATION (ή GPE αν δεν έχεις LOCATION). Αν ενεργεί ("αποφάσισε"), είναι GPE.

        Text: "{text}"
        
        JSON Format: {{ "entities": [ {{ "text": "...", "label": "..." }} ] }}
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://legal-annotator.local", 
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a precise NER system. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            # Llama usually does not support 'json_object' response_format enforced by API universally yet on OpenRouter, 
            # but usually respects instructions. We omit strict mode to be safe or use if supported.
            # We will try standard prompting first.
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Basic cleanup
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                data = json.loads(content)
                valid = []
                for ent in data.get('entities', []):
                    if ent.get('text'):
                        ent['source'] = 'CloudScanner (Llama 3.3)'
                        valid.append(ent)
                return valid
        except Exception as e:
            # print(f"Scanner error: {e}")
            pass
        return []


class OllamaClient:
    """
    Tier 0 Enhancer: Local LLM (Meltemi / Llama 3).
    Runs completely offline via Ollama.
    """
    def __init__(self, model_name="meltemi"):
        # Default to meltemi, fallback to llama3 if user specifies
        self.model = model_name
        self.api_url = "http://localhost:11434/api/chat"
        self.is_ready = self._check_connection()
        
    def _check_connection(self):
        try:
            # Quick check to see if Ollama is running
            response = requests.get("http://localhost:11434/", timeout=2)
            if response.status_code == 200:
                print(f"✅ Ollama detected locally.")
                return True
        except:
             print(f"⚠️ Ollama is NOT running on localhost:11434.")
        return False
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Uses Meltemi's Greek capabilities to find entities locally.
        """
        if not self.is_ready:
            return []
            
        system_prompt = """
        Είσαι ειδικός στην εξαγωγή νομικών οντοτήτων από Ελληνικά κείμενα.
        Εντόπισε ΟΛΕΣ τις οντότητες στο κείμενο.
        
        Κατηγορίες:
        - ORG (Οργανισμοί, Εταιρείες, Υπουργεία)
        - GPE (Πόλεις, Χώρες, Δήμοι)
        - PERSON (Φυσικά Πρόσωπα)
        - FACILITY (Κτίρια, Υποδομές)
        - LEG_REFS (Νόμοι: 'ν. 4000/2012', 'Π.Δ. 80/2020')
        - PUBLIC_DOCS (ΦΕΚ, Αποφάσεις: 'ΦΕΚ Α 80')
        - LOCATION (Γεωγραφικοί τόποι)
        - DATE (Ημερομηνίες)
        
        Απάντησε ΜΟΝΟ με JSON μορφή:
        {
            "entities": [
                {"text": "...", "label": "..."},
                ...
            ]
        }
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "stream": False,
            "format": "json" # Ollama supports strict JSON mode
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                content = data.get('message', {}).get('content', '{}')
                result = json.loads(content)
                
                valid = []
                for ent in result.get('entities', []):
                    # Ensure text is in the sentence
                    if ent.get('text') and ent.get('label'):
                         # Basic fuzzy check or exact check
                         if ent['text'] in text:
                             ent['source'] = f'Local LLM ({self.model})'
                             valid.append(ent)
                return valid
        except Exception as e:
            # print(f"⚠️ Ollama ({self.model}) Error: {e}")
            pass
            return []

