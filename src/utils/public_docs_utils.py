import re

def find_public_docs_title_span(text, entity_start, entity_end, window=10):
    """
    Εντοπίζει τον τίτλο δημόσιου εγγράφου (PUBLIC-DOCS) κοντά στο entity span.
    - Ψάχνει για τίτλο σε εισαγωγικά, παρενθέσεις ή μετά από λέξεις-κλειδιά (απόφαση, ΦΕΚ, αριθμ.).
    - Επιστρέφει (start, end) indices του τίτλου ή (entity_start, entity_end) αν δεν βρεθεί τίτλος.
    """
    # Ορισμοί
    keywords = [
        r'απόφαση', r'ΦΕΚ', r'αριθμ\.', r'Αριθμ\.', r'Πολ\.', r'Εγκύκλιος', r'Σύμβαση', r'Πράξη', r'Πρωτόκολλο', r'Διακήρυξη'
    ]
    quote_patterns = [
        r'«([^»]{3,100})»',
        r'“([^”]{3,100})”',
        r'"([^"]{3,100})"',
        r'\(([^\)]{3,100})\)'
    ]
    # Πάρε tokens και offsets
    tokens = list(re.finditer(r'\S+', text))
    entity_token_idx = None
    for i, m in enumerate(tokens):
        if m.start() <= entity_start < m.end():
            entity_token_idx = i
            break
    if entity_token_idx is None:
        return (entity_start, entity_end)
    # Ψάξε σε παράθυρο γύρω από το entity
    start_idx = max(0, entity_token_idx - window)
    end_idx = min(len(tokens), entity_token_idx + window + 1)
    window_text = text[tokens[start_idx].start():tokens[end_idx-1].end()]
    window_offset = tokens[start_idx].start()
    # 1. Ψάξε για εισαγωγικά/παρενθέσεις
    for pat in quote_patterns:
        for m in re.finditer(pat, window_text):
            span_start = window_offset + m.start(1)
            span_end = window_offset + m.end(1)
            # Αν ο τίτλος είναι κοντά στο entity, επέστρεψέ το (ενωμένο με το entity)
            if abs(span_start - entity_end) < 40 or abs(span_end - entity_start) < 40:
                return (min(entity_start, span_start), max(entity_end, span_end))
    # 2. Ψάξε για keywords και πάρε 2-10 λέξεις μετά
    for kw in keywords:
        for m in re.finditer(kw, window_text, re.IGNORECASE):
            after = window_text[m.end():].strip()
            after_match = re.match(r'([\S\s]{0,80})', after)
            if after_match:
                after_text = after_match.group(1)
                # Πάρε μέχρι τελεία/εισαγωγικά/παρένθεση/αριθμό
                enders = re.search(r'[»."\)\n]', after_text)
                if enders:
                    span_end = window_offset + m.end() + enders.start()
                else:
                    span_end = window_offset + m.end() + len(after_text)
                span_start = window_offset + m.end()
                # Επιστροφή αν κοντά στο entity (ενωμένο)
                if abs(span_start - entity_end) < 40 or abs(span_end - entity_start) < 40:
                    return (min(entity_start, span_start), max(entity_end, span_end))
    # Αν δεν βρέθηκε τίτλος, επέστρεψε το αρχικό span
    return (entity_start, entity_end)
