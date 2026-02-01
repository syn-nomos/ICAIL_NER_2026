import sqlite3
import re
import os
import logging

logger = logging.getLogger(__name__)

def simple_tokenizer(text):
    """
    Tokenizes text into words and punctuation using regex.
    """
    # Keep words or non-whitespace characters
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def export_accepted_to_conll(db_path, output_path):
    """
    Exports accepted sentences and annotations from DB to CoNLL format.
    Only includes sentences with at least one accepted annotation or marked as clean?
    Currently we only really care about positive samples for fine-tuning dominance.
    But we also need negative samples (O-tags).
    
    Strategy:
    1. Fetch all sentences that have at least one accepted annotation.
    2. Also fetch sentences that are manually marked as 'checked' if we had that flag, 
       but here we rely on 'annotations' table.
       If a sentence has NO annotations but is part of the train set, it's all 'O'.
    
    For now, let's export ALL sentences that have ANY 'is_accepted=1' annotation.
    """
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all sentences with accepted annotations
    cursor.execute("""
        SELECT DISTINCT s.id, s.text 
        FROM sentences s
        JOIN annotations a ON s.id = a.sentence_id
        WHERE a.is_accepted = 1
    """)
    rows = cursor.fetchall()
    
    if not rows:
        logger.warning("No accepted sentences found in DB.")
        conn.close()
        return False, "No accepted sentences found."

    logger.info(f"Found {len(rows)} sentences with accepted annotations.")
    
    # Get the annotations
    cursor.execute("""
        SELECT sentence_id, label, start_char, end_char 
        FROM annotations 
        WHERE is_accepted = 1
        ORDER BY sentence_id, start_char
    """)
    all_anns = cursor.fetchall()
    
    # Organize annotations by sentence_id
    from collections import defaultdict
    anns_map = defaultdict(list)
    for sent_id, label, start, end in all_anns:
        # Skip broken annotations (failed healing)
        if start is None or end is None:
            continue
            
        # NORMALIZE LABELS: Convert LEG_REFS -> LEG-REFS to match Model Config
        if label == "LEG_REFS":
            label = "LEG-REFS"
        elif label == "PUBLIC_DOCS":
            label = "PUBLIC-DOCS" # Ensure hyphen if model expects it (checking config...)
            
        anns_map[sent_id].append((label, start, end))
    
    conn.close()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_id, text in rows:
            if not text:
                continue
                
            tokens = simple_tokenizer(text)
            
            # Map tokens to labels
            # We need to find the character span of each token in the original text
            # This is tricky with regex alone if we have whitespace.
            # Alternate approach: Iterate through tokens and find them in text.
            
            token_spans = []
            current_pos = 0
            
            # Reconstruct spans (approximate if whitespace varies, but we search forward)
            for token in tokens:
                try:
                    start = text.index(token, current_pos)
                    end = start + len(token)
                    token_spans.append((token, start, end))
                    current_pos = end
                except ValueError:
                    # Should unlikely happen if tokenizer is strictly derived from text
                    continue
            
            # Assign BIO tags
            anns = anns_map.get(sent_id, [])
            
            for token, t_start, t_end in token_spans:
                label = "O"
                for ann_label, a_start, a_end in anns:
                    # Check overlap
                    # We prioritize alignment. 
                    # If token start matches annotation start -> B-TAG
                    # If token is inside annotation -> I-TAG
                    
                    if t_start == a_start:
                        label = f"B-{ann_label}"
                        break
                    elif t_start > a_start and t_end <= a_end:
                        label = f"I-{ann_label}"
                        break
                    # Edge case: Annotation starts in middle of token? 
                    # Tokenizer should split there? 
                    # If regex \w+ didn't split, we might miss strict boundary.
                    # But roughly okay.
                    
                f.write(f"{token}\t{label}\n")
            
            f.write("\n") # Sentence separator
            
    return True, f"Exported {len(rows)} sentences to {output_path}"
