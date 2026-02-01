
import re

def clean_public_docs_match(text):
    """
    Cleans up PUBLIC-DOCS matches by removing surrounding "noise" words like "απόφαση" 
    that might have been captured by loose regex.
    """
    if not text: return ""
    
    # 1. Remove leading words "απόφαση", "εγκύκλιος" etc if they are at the very start
    # We want to keep the Reference (Number/Year) and the Title.
    # Pattern: ^(απόφαση\s+|υπουργικής\s+απόφασης\s+)(.*)$
    # Note: Regex in Python is case sensitive unless flag is set, but we usually normalize before specific checks or use flags.
    # Here we are cleaning the *span* text returned by regex.
    
    clean = text
    
    # List of prefixes to strip
    prefixes = [
        r"^(?:την\s+)?(?:υπ[\.\']?\s*)?(?:αριθμ[\.\']?\s*)?(?:πρωτ[\.\']?\s*)?απόφαση(?:ς)?\s+",
        r"^(?:την\s+)?(?:υπ[\.\']?\s*)?(?:αριθμ[\.\']?\s*)?(?:πρωτ[\.\']?\s*)?εγκύκλιος(?:ς)?\s+",
        r"^(?:την\s+)?(?:υπ[\.\']?\s*)?(?:αριθμ[\.\']?\s*)?(?:πρωτ[\.\']?\s*)?πράξη(?:ς)?\s+",
        r"^(?:την\s+)?(?:υπ[\.\']?\s*)?αριθμ\.\s+",
        r"^(?:την\s+)?(?:υπ[\.\']?\s*)?αριθ\.\s+",
        r"^(?:την\s+)?(?:υπ[\.\']?\s*)?αρ\.\s+",
        r"^της\s+",
        r"^του\s+",
        r"^την\s+",
    ]
    
    for p in prefixes:
        # Use simple sub just for the start of the string
        clean = re.sub(p, "", clean, flags=re.IGNORECASE | re.UNICODE)
        
    return clean.strip()
