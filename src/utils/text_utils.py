def find_quote_span(text, start, end):
    """
    Επιστρέφει (new_start, new_end) ώστε το span να περιλαμβάνει όλο το περιεχόμενο ανάμεσα σε σύμβολα τίτλου.
    Προτεραιότητα:
    1. Έλεγχος δεξιά (μετά το end) για τίτλο σε εισαγωγικά (π.χ. Νόμος 1234 «Τίτλος»).
    2. Έλεγχος αριστερά/δεξιά για τίτλο που περικλείει το entity (π.χ. «Νόμος 1234»).
    """
    # Ορισμοί συμβόλων
    left_quotes = ["«", "\u201C", "\u00AB", "\u2039", "\u2018", "\u201E", "\u201F", "\u275B", "\u275D", "\u301D", "\u301F", "\u2E42", "\u300A", "\u3008", "\u300C", "\u300E", "\u3010", "\u3014", "\u3016", "\u3018", "\u301A", "\u301C", "\u301E", "\u2E02", "\u2E04", "\u2E09", "\u2E0C", "\u2E1C", "\u2E20", "\u00AB", "\u2039", "\u2329", "\u3008", "\u300A", "\u300C", "\u300E", "\u3010", "\u3014", "\u3016", "\u3018", "\u301A", "\u301C", "\u301E", "\u2E02", "\u2E04", "\u2E09", "\u2E0C", "\u2E1C", "\u2E20", "\u00AB", "\u2039", "\u2329", "\u3008", "\u300A", "\u300C", "\u300E", "\u3010", "\u3014", "\u3016", "\u3018", "\u301A", "\u301C", "\u301E", "\u2E02", "\u2E04", "\u2E09", "\u2E0C", "\u2E1C", "\u2E20", "\u00AB", "\u2039", "\u2329", "\u3008", "\u300A", "\u300C", "\u300E", "\u3010", "\u3014", "\u3016", "\u3018", "\u301A", "\u301C", "\u301E", "\u2E02", "\u2E04", "\u2E09", "\u2E0C", "\u2E1C", "\u2E20", "\u00AB", "\u2039", "\u2329", "\u3008", "\u300A", "\u300C", "\u300E", "\u3010", "\u3014", "\u3016", "\u3018", "\u301A", "\u301C", "\u301E", "\u2E02", "\u2E04", "\u2E09", "\u2E0C", "\u2E1C", "\u2E20", "<<", "\"", "“"]
    right_quotes = ["»", "\u201D", "\u00BB", "\u203A", "\u2019", "\u201C", "\u201F", "\u275C", "\u275E", "\u301E", "\u301F", "\u2E03", "\u2E05", "\u2E0A", "\u2E0D", "\u2E1D", "\u2E21", "\u232A", "\u3009", "\u300B", "\u300D", "\u300F", "\u3011", "\u3015", "\u3017", "\u3019", "\u301B", "\u301D", "\u301F", ">>", "\"", "”"]
    
    # --- 1. Search Right (Forward) ---
    first_lq_idx = -1
    first_lq_len = 0
    search_limit = min(len(text), end + 50)
    
    for lq in left_quotes:
        idx = text.find(lq, end, search_limit)
        if idx != -1:
            if first_lq_idx == -1 or idx < first_lq_idx:
                first_lq_idx = idx
                first_lq_len = len(lq)
                
    if first_lq_idx != -1:
        first_rq_idx = -1
        first_rq_len = 0
        for rq in right_quotes:
            idx = text.find(rq, first_lq_idx + first_lq_len)
            if idx != -1:
                if idx - first_lq_idx < 400:
                    if first_rq_idx == -1 or idx < first_rq_idx:
                        first_rq_idx = idx
                        first_rq_len = len(rq)
                        
        if first_rq_idx != -1:
            return start, first_rq_idx + first_rq_len

    # --- 2. Search Surrounding (Backward) ---
    last_lq_idx = -1
    last_lq_len = 0
    for lq in left_quotes:
        idx = text.rfind(lq, 0, start)
        if idx > last_lq_idx:
            last_lq_idx = idx
            last_lq_len = len(lq)
            
    if last_lq_idx != -1:
        is_closed = False
        for rq in right_quotes:
            c_idx = text.find(rq, last_lq_idx + last_lq_len, start)
            if c_idx != -1:
                is_closed = True
                break
        
        if not is_closed:
            first_rq_idx = -1
            first_rq_len = 0
            for rq in right_quotes:
                idx = text.find(rq, end)
                if idx != -1:
                     if first_rq_idx == -1 or idx < first_rq_idx:
                        first_rq_idx = idx
                        first_rq_len = len(rq)
            
            if first_rq_idx != -1:
                return last_lq_idx, first_rq_idx + first_rq_len

    return start, end

import unicodedata
import re
import html as html_lib
import streamlit as st

def normalize_text(text: str) -> str:
    """
    Removes Greek accents and converts to lowercase.
    Example: "Άρειος Πάγος" -> "αρειος παγος"
    """
    if not text:
        return ""
    
    # Normalize unicode characters (NFD splits characters from accents)
    normalized = unicodedata.normalize('NFD', text)
    
    # Filter out non-spacing mark characters (accents)
    no_accents = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Convert to lowercase
    return no_accents.lower()

def get_pseudo_stem(word: str) -> str:
    """
    Simple heuristic to get the 'root' of a Greek word to handle cases.
    Removes common endings like 'ος', 'ου', 'ας', 'α', 'ων', 'ες', 'οι'.
    This is not a full stemmer, but good enough for SQL LIKE queries.
    """
    w = normalize_text(word)
    if len(w) < 4:
        return w
        
    # Common 2-letter endings to strip
    endings = ['ος', 'ου', 'ας', 'ων', 'ες', 'οι', 'ης', 'ου', 'α', 'η', 'ο', 'ι']
    # Sort by length descending to match longest first
    endings.sort(key=len, reverse=True)
    
    for end in endings:
        if w.endswith(end):
            return w[:-len(end)]
            
    return w

def get_entity_color(label):
    colors = {
        'ORG': '#e3f2fd',      # Light Blue
        'PERSON': '#f1f8e9',   # Pale Green
        'GPE': '#fff8e1',      # Amber/Cream
        'FACILITY': '#f3e5f5', # Pale Purple
        'DATE': '#e0f7fa',     # Cyan
        'LOCATION': '#fffde7', # Light Yellow
        # Distinct Colors for Legal Refs vs Public Docs
        'LEG_REFS': '#ffcdd2', # Reddish Pink (Danger/Law)
        'PUBLIC_DOCS': '#cfd8dc', # Blue-Grey (Neutral Docs)
        'LEG-REFS': '#ffcdd2', # Alias for safety
        'PUBLIC-DOCS': '#cfd8dc' # Alias for safety
    }
    return colors.get(label, '#f5f5f5')

def get_entity_border(label):
    colors = {
        'ORG': '#2196f3',      # Blue
        'PERSON': '#4caf50',   # Green
        'GPE': '#ff9800',      # Orange
        'FACILITY': '#9c27b0', # Purple
        'DATE': '#00bcd4',     # Cyan
        'LOCATION': '#ffc107', # Yellow
        # Stronger Borders
        'LEG_REFS': '#b71c1c', # Deep Red
        'PUBLIC_DOCS': '#455a64', # Dark Blue Grey
        'LEG-REFS': '#b71c1c', # Alias
        'PUBLIC-DOCS': '#455a64' # Alias
    }
    return colors.get(label, '#9e9e9e')

def highlight_sentence(text, annotations):
    if not annotations:
        return html_lib.escape(text)
        
    # Sort annotations by start_char (asc) and then by length (desc) to prefer longer spans
    sorted_anns = sorted(annotations, key=lambda x: (x['start_char'], -(x['end_char'] - x['start_char'])))
    
    result = ""
    last_idx = 0
    
    for i, ann in enumerate(sorted_anns):
        start = ann['start_char']
        end = ann['end_char']
        label = ann['label']
        
        # Safety clamps
        if start < 0: start = 0
        if end > len(text): end = len(text)
        if start > end: continue # Invalid span

        # Safety check overlap
        if start < last_idx: continue # Skip overlap
        
        # Append text before annotation
        result += html_lib.escape(text[last_idx:start])
        
        # Determine style
        bg_color = get_entity_color(label)
        border_color = get_entity_border(label)
        
        # Add badge if it's a pending annotation
        badge_html = ""
        if 'display_index' in ann:
            badge_html = f"<span class='entity-badge'>#{ann['display_index']}</span>"
            
        # Append highlighted span
        content = html_lib.escape(text[start:end])
        result += f"<span class='entity-highlight' style='background-color:{bg_color}; border-color:{border_color};' title='{label}'>{badge_html}{content}</span>"
        
        last_idx = end
        
    result += html_lib.escape(text[last_idx:])
    return result
    return result

def adjust_boundaries(text, start, end, action):
    """
    Adjusts start/end indices based on action:
    - expand_left: Move start left to include previous word.
    - shrink_left: Move start right to exclude current first word.
    - expand_right: Move end right to include next word.
    - shrink_right: Move end left to exclude current last word.
    """
    s, e = start, end
    
    if action == "expand_left":
        # Move left skipping whitespace, then move left until whitespace or start
        curr = s - 1
        while curr >= 0 and text[curr].isspace(): curr -= 1
        while curr >= 0 and not text[curr].isspace(): curr -= 1
        s = curr + 1
        
    elif action == "shrink_left":
        # Move right until whitespace, then skip whitespace
        curr = s
        while curr < e and not text[curr].isspace(): curr += 1
        while curr < e and text[curr].isspace(): curr += 1
        if curr < e: s = curr
            
    elif action == "expand_right":
        # Move right skipping whitespace, then move right until whitespace or end
        curr = e
        while curr < len(text) and text[curr].isspace(): curr += 1
        while curr < len(text) and not text[curr].isspace(): curr += 1
        e = curr
        
    elif action == "shrink_right":
        # Move left until whitespace, then skip whitespace
        curr = e - 1
        while curr > s and text[curr].isspace(): curr -= 1
        while curr > s and not text[curr].isspace(): curr -= 1
        if curr >= s: e = curr + 1 # +1 because end is exclusive
            
    return s, e

@st.cache_data
def render_tokenized_text(text):
    """Renders text broken down by words with Start/End character indices."""
    tokens = list(re.finditer(r'\S+', text))
    # Use a flex container for better wrapping and spacing
    html_parts = ["<div style='line-height: 1.8; background-color:#fff; padding:15px; border-radius:5px; border:1px solid #eee; font-family: sans-serif;'>"]
    
    for token in tokens:
        s, e = token.span()
        word = html_lib.escape(token.group())
        token_html = (
            f"<div class='token-container'>"
            f"<span class='idx-badge-start'>{s}</span>"
            f"<span class='token-word'>{word}</span>"
            f"<span class='idx-badge-end'>{e}</span>"
            f"</div>"
        )
        html_parts.append(token_html)
        
    html_parts.append("</div>")
    return " ".join(html_parts)
