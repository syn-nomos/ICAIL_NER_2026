
import json
import re

PROMPTS = {
    "PERSON": """
    You are an expert Greek linguist specializing in Named Entity Recognition. Your goal is to extract the Full Name of a natural person from specific context, correcting boundaries.

    ### CRITICAL CONTEXT INSTRUCTION:
    The provided 'Target Text' might be an incomplete fragment. YOU MUST LOOK AT THE SURROUNDING 'SENTENCE CONTEXT' provided.
    If the name continues before or after the target text (e.g. includes a patronym like '... του Ιωάννη'), EXTEND the selection to include it.
    
    ### RULES:
    1. **DO NOT CHOP WORDS**: 
       - NEVER output a cutoff name like "ΡΙΚΩΝ" (for "ΕΣΩΤΕΡΙΚΩΝ") or "ΥΠΟΔΟΜΩΝ" (if incomplete).
       - Always include the FULL word.
       
    2. **REMOVE LEADING ARTICLES/TITLES**: 
       - Delete words like: "ο", "η", "του", "της", "τον", "την", "κ.", "κα", "κυριος", "κυρία", "Δρ.", "Prof.".
       - Example: "του Γεωργίου Παπαδόπουλου" -> "Γεωργίου Παπαδόπουλου"
       - Example: "η κα. Μαρία" -> "Μαρία"
    
    3. **KEEP PATRONYMS (Middle/End)**:
       - If the name includes "του [Name]" (son of ...) AS PART OF THE IDENTITY, keep it.
       - Example: "Μαρία Γρηγορίου του Μανώλη" -> KEPT AS IS: "Μαρία Γρηγορίου του Μανώλη" (Do not remove the internal 'του').
       - Example: "Ιωάννης Παππάς του Κωνσταντίνου" -> KEPT AS IS.
    
    3. **REMOVE PUNCTUATION**:
       - Remove trailing commas, dots, or quotes that are not part of the name.
       - Example: "Γεώργιος," -> "Γεώργιος"

    ### EXAMPLES:
    Input Target: "Γεωργίου"
    Context: "...από τον κ. Γεωργίου Παπαδόπουλου του Ιωάννη..."
    Output: "Γεωργίου Παπαδόπουλου του Ιωάννη"
    
    Input Target: "της Ελένης"
    Context: "της Ελένης Κ. του Δημητρίου,"
    Output: "Ελένης Κ. του Δημητρίου"
    """,

    "ORG": """
    You are an expert Greek legal annotator. Extract the OFFICIAL ORGANIZATION NAME (Companies, Ministries, Institutions).
    
    ### CRITICAL CONTEXT INSTRUCTION:
    The provided 'Target Text' might be a fragment of a longer name. Look at the surrounding context!
    ESPECIALLY for Ministries (Υπουργεία) that are joined (e.g. "Υπουργείο Οικονομίας και Ανάπτυξης"). 
    If you see "...και..." linking two departments or names, INCLUDE BOTH parts if they form a single entity name.
    
    ### RULES:
    1. **DO NOT CHOP WORDS**: 
       - NEVER output a cutoff name. Always include the FULL word if visible in context.
       
    2. **KEEP LEGAL SUFFIXES**:
       - Must include: "Α.Ε.", "Ε.Π.Ε.", "Ι.Κ.Ε.", "Ltd", "S.A.", "Corp", "Ο.Ε.", "Ε.Ε.".
       - Example: "Info Quest Technologies Α.Ε." -> Keep all.
    
    2. **REMOVE LEADING ARTICLES/PREPOSITIONS**:
       - Remove: "η", "την", "στην", "από την", "του", "της".
       - Example: "στην εταιρεία ΤΕΡΝΑ Α.Ε." -> "ΤΕΡΝΑ Α.Ε." (Remove 'στην εταιρεία' only if generic).

    3. **HANDLE COMPLEX MINISTRY NAMES**:
       - Multi-part names joined by "και" must be kept together.
       - Example Target: "Ναυτιλίας"
       - Context: "...Υπουργείο Ναυτιλίας και Αιγαίου..."
       - Output: "Υπουργείο Ναυτιλίας και Αιγαίου"
    
    ### EXAMPLES:
    Input Target: "ΜΥΤΙΛΗΝΑΙΟΣ"
    Context: "με την ΜΥΤΙΛΗΝΑΙΟΣ Α.Ε. του ομίλου"
    Output: "ΜΥΤΙΛΗΝΑΙΟΣ Α.Ε."
    
    Input Target: "Αιγαίου"
    Context: "...καθώς και Ναυτιλίας και Αιγαίου την εκτέλεσή..."
    (Implied context: Likely "Υπουργείο ...")
    Output: "Ναυτιλίας και Αιγαίου" (or full "Υπουργείο Ναυτιλίας και Αιγαίου" if visible in context).
    """,

    "GPE": """
    Extract the GEOPOLITICAL ENTITY (City, Country, Region, Municipality).
    
    ### CRITICAL CONTEXT INSTRUCTION:
    Look for multi-word locations (e.g. "San Francisco", "Νέα Υόρκη", "Άγιος Νικόλαος").
    If the target is just "Νέα" or "Υόρκη", expand to "Νέα Υόρκη".
    
    ### RULES:
    1. **REMOVE PREPOSITIONS**:
       - Remove "σε", "από", "στην", "στο", "του", "της" (if possessive of sentence).
       - Example: "στο Δήμο Αθηναίων" -> "Δήμο Αθηναίων"
       - Preferred: "Δήμος Αθηναίων" over "Αθηναίων".
    
    2. **CLEANING**:
       - Remove surrounding punctuation.

    ### EXAMPLES:
    Input Target: "Θεσσαλονίκης"
    Context: "κάτοικος Θεσσαλονίκης"
    Output: "Θεσσαλονίκης"
    """,
    
    "DATE": """
    Extract the EXACT DATE expression in a standalone strings.
    
    ### RULES:
    1. **FORMAT**: Keep Day, Month, Year.
    2. **CLEAN**: Remove "στις", "την", "από", "ημερομηνία".
    3. **NUMERIC or TEXT**: Keep both "1/1/2020" and "1η Ιανουαρίου 2020".
    """,

    "FACILITY": """
    Extract the full name of the FACILITY (Roads, Buildings, Infrastructure).
    
    ### CRITICAL CONTEXT INSTRUCTION:
    Include the FULL proper name, including Type (Οδός, Λεωφόρος).
    If target is "Κηφισίας", Context is "επί της Λεωφόρου Κηφισίας", Output "Λεωφόρου Κηφισίας".
    
    ### RULES:
    1. **INCLUDE TYPE**: Keep "Οδός", "Λεωφόρος", "Γέφυρα", "Κτίριο".
    2. **REMOVE PREPOSITIONS**: Remove "επί της", "στην".
    """,
    
    "LEG-REFS": """
    Extract the LEGAL REFERENCE (Law, Act, Article).
    
    ### RULES:
    1. **FORMAT**: Must be [Type] [Number]/[Year].
    2. **TYPES**: Ν. (Νόμος), Π.Δ., Κ.Υ.Α., Υ.Α., άρθρο.
    3. **CONTEXT**: Remove "σύμφωνα με", "του", "βάσει".
    """,

    "PUBLIC-DOCS": """
    Extract the PUBLIC DOCUMENT reference (Protocol Number, Decision ID, ADA).
    
    ### RULES:
    1. Keep the whole ID string (Numbers, slashes, descriptors).
    2. Example: "ΑΔΑ: 6ΨΩΗ4653Π4-ΥΡΓ".
    """,
    
    "LOCATION": """
    Extract NATURAL LOCATIONS (Mountains, Rivers, Continents, Areas).
    Remove "στον", "στον ποταμό" (keep just name if generic) or keep "Ποταμός Χ" if it is the name.
    """
}

class TypeSpecificBoundaryFixer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def fix_boundary(self, text_context, rough_span, label):
        """
        Uses LLM to refine the boundary of 'rough_span' inside 'text_context'
        based on the 'label' specific rules.
        """
        if label not in PROMPTS:
            label = "ORG" # Fallback or Generic
            
        system_prompt = PROMPTS.get(label, PROMPTS["ORG"])
        
        user_prompt = f"""
        Sentence context: "...{text_context}..."
        Target Text to Fix: "{rough_span}"
        
        Task: Identify the exact logical boundary of the entity strictly according to the rules.
        Return a JSON object with a single key "fixed_text" containing the corrected string.
        Example: {{ "fixed_text": "Corrected Entity Name" }}
        """
        
        # Use the robust LLM client (with retries, long timeout, and cleaning)
        response_json = self.llm._call_llm(system_prompt, user_prompt)
        
        if response_json and isinstance(response_json, dict) and "fixed_text" in response_json:
            return str(response_json["fixed_text"]).strip()
            
        return rough_span
            

