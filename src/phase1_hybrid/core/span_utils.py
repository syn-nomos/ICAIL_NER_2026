from typing import List, Dict, Any

def group_tokens_to_spans(text: str, token_preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans = []
    current_entity = None
    current_offset = 0
    
    for item in token_preds:
        token = item['token']
        label = item['label']
        confidence = item['confidence']
        # ΝΕΟ: Παίρνουμε το embedding αν υπάρχει
        embedding = item.get('embedding')
        
        clean_token = token.replace('Ġ', '').replace('##', '')
        start_char = text.find(clean_token, current_offset)
        
        if start_char == -1 or clean_token in ["<cls>", "<sep>", "[CLS]", "[SEP]"]:
            continue

        end_char = start_char + len(clean_token)
        
        if label.startswith('B-'):
            if current_entity: spans.append(current_entity)
            
            entity_type = label.split('-')[1]
            current_entity = {
                'start': start_char,
                'end': end_char,
                'label': entity_type,
                'text': clean_token,
                'confidence_sum': confidence,
                'token_count': 1,
                'source': 'RoBERTa',
                'token_embeddings': [embedding] if embedding is not None else [] # Λίστα vectors
            }
        
        elif label.startswith('I-') and current_entity:
            entity_type = label.split('-')[1]
            if entity_type == current_entity['label']:
                current_entity['end'] = end_char
                current_entity['text'] = text[current_entity['start']:current_entity['end']]
                current_entity['confidence_sum'] += confidence
                current_entity['token_count'] += 1
                if embedding is not None:
                    current_entity['token_embeddings'].append(embedding)
            else:
                spans.append(current_entity)
                current_entity = None
                
        else:
            if current_entity: spans.append(current_entity)
            current_entity = None
        
        current_offset = end_char 
        
    if current_entity: spans.append(current_entity)

    # Use the shared function
    spans = snap_entities_to_words(spans, text)
    spans = merge_overlapping_spans(spans, text)
    return spans

def snap_entities_to_words(entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Refines entity boundaries to align with full words.
    Fixes cases like "iltiadis" -> "Miltiadis" (RoBERTa subword issue).
    """
    final_spans = []
    text_len = len(text)
    
    for span in entities:
        # Calculate scores if pending
        if 'token_count' in span and span['token_count'] > 0:
            span['confidence'] = span['confidence_sum'] / span['token_count']
            del span['confidence_sum']
            del span['token_count']
            
        # Ensure text is present in span if missing
        orig_start = span['start']
        orig_end = span['end']
        
        # If text is missing or mismatched length (stale), refresh it from source text
        # This protects against spans created without text field
        if 'text' not in span:
            span['text'] = text[orig_start:orig_end]
        
        # Expand Logic
        
        # 1. Expand LEFT: Move start back if previous char is ALPHANUMERIC
        new_start = orig_start
        # Safety limit: Don't go back more than 50 chars (avoids infinite loops in weird text)
        limit = 0
        while new_start > 0 and text[new_start-1].isalnum() and limit < 50:
            new_start -= 1
            limit += 1
            
        # 2. Expand RIGHT: Move end forward if next char is ALPHANUMERIC
        new_end = orig_end
        limit = 0
        while new_end < text_len and text[new_end].isalnum() and limit < 50:
            new_end += 1
            limit += 1
            
        # Update if changed
        if new_start != orig_start or new_end != orig_end:
            span['start'] = new_start
            span['end'] = new_end
            span['text'] = text[new_start:new_end]
            # Mark as auto-corrected
            span['auto_snapped'] = True
            
        final_spans.append(span)
        
    return final_spans

def merge_overlapping_spans(entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Merges entities that overlap and have the same label.
    Useful after snapping, which might cause fragments to expand into the same word.
    """
    if not entities:
        return []
        
    # Sort by start position
    sorted_ents = sorted(entities, key=lambda x: (x['start'], x['end']))
    merged = []
    
    current = sorted_ents[0]
    
    for next_ent in sorted_ents[1:]:
        # Check overlap
        # Overlap exists if next_ent['start'] < current['end']
        # We assume they are sorted by start, so next_ent['start'] >= current['start']
        
        overlap = next_ent['start'] < current['end']
        same_label = next_ent['label'] == current['label']
        
        if overlap and same_label:
            # Merge
            new_start = min(current['start'], next_ent['start'])
            new_end = max(current['end'], next_ent['end'])
            
            # Keep max confidence
            new_conf = max(current.get('confidence', 0), next_ent.get('confidence', 0))
            
            current['start'] = new_start
            current['end'] = new_end
            current['text'] = text[new_start:new_end]
            current['confidence'] = new_conf
            # Inherit source if mixed? keep current.
        else:
            merged.append(current)
            current = next_ent
            
    merged.append(current)
    return merged