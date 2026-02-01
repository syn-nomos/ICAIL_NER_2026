import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class TriggerInfo:
    text: str
    score: float
    direction: str  # 'LEFT' or 'RIGHT' (or 'INSIDE' typically filtered)
    tokens: List[str]

class AttentionExtractor:
    """
    Extracts 'Trigger Words' using Self-Attention.
    - Reconstructs full words from subword tokens.
    - Identifies direction (LEFT/RIGHT) relative to entity.
    - Aggregates attention scores per word.
    """
    
    def __init__(self, stopword_list=None):
        self.stopwords = stopword_list or {
            'και', 'η', 'ο', 'το', 'τα', 'του', 'της', 'των', 'τον', 'την', 'τους', 'τις', 
            'με', 'σε', 'για', 'από', 'προς', 'κατά', 'ότι', 'πως', 'που', 'είναι', 'ήταν',
            'αυτό', 'αυτή', 'εκ', 'επί', 'έως', 'μέχρι', 'δεν', 'μην', 'σαν', 'ως'
        }
        self.punctuation = set('.,;:-db"\'()[]{}«»')
        # SentencePiece Start-of-word marker (U+2581)
        self.sp_marker = '\u2581' 

    def is_valid_word(self, word: str) -> bool:
        """Filters out stopwords, punctuation, and short nonsense."""
        if not word: return False
        if word in {'<s>', '</s>', '<pad>', '<unk>', '<mask >'}: return False
        if len(word) < 2: return False # Relaxed from 3 to 2 for short prepositions if needed, though stopwords filter them
        if word in self.stopwords: return False
        if all(char in self.punctuation for char in word): return False
        if word.isdigit(): return False
        return True

    def extract_triggers(self, roberta_wrapper, text: str, entity_start_char: int, entity_end_char: int, top_k=3) -> List[TriggerInfo]:
        """
        Returns a list of TriggerInfo objects with consolidated word scores and direction.
        """
        tokenizer = roberta_wrapper.tokenizer
        model = roberta_wrapper.model
        device = roberta_wrapper.device

        # 1. Tokenize & Run Model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
        input_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 2. Get Attention (Last Layer, Average Heads)
        last_layer_attn = outputs.attentions[-1][0]
        avg_attn = last_layer_attn.mean(dim=0).cpu().numpy()
        
        # 3. Identify Entity Indices
        entity_indices = []
        for i, (o_start, o_end) in enumerate(offsets):
            if o_start == 0 and o_end == 0: continue
            overlap_start = max(entity_start_char, o_start)
            overlap_end = min(entity_end_char, o_end)
            if overlap_end > overlap_start:
                entity_indices.append(i)
                
        if not entity_indices: return []

        # Center of entity (for LEFT/RIGHT logic)
        ent_center = sum(entity_indices) / len(entity_indices)

        # 4. Score Tokens based on how much Entity attends to them
        token_scores = avg_attn[entity_indices, :].mean(axis=0)

        # 5. RECONSTRUCT WORDS & AGGREGATE SCORES
        words: List[TriggerInfo] = []
        
        current_word_parts = []
        current_word_score = 0.0
        current_word_start_idx = -1
        
        for i, token in enumerate(tokens):
            score = float(token_scores[i])
            
            # Check if this token starts a new word
            # Usually marked by Ġ or starts with unicode SPIECE marker
            # We explicitly check for both common markers
            is_start = token.startswith('Ġ') or token.startswith(self.sp_marker) or token.startswith(' ')
            
            # Special case: First word might not have marker depending on tokenizer
            if i == 0: is_start = True 
            # Punctuation often treated as start of new token in some tokenizers logic, but let's stick to markers
            
            if is_start:
                # Flush previous word
                if current_word_parts:
                    # Clean up special tokens explicitly here
                    raw_text = "".join(current_word_parts)
                    clean_text = raw_text.replace('Ġ', '').replace(self.sp_marker, '').replace(' ', '')
                    clean_text = clean_text.replace('</s>', '').replace('<s>', '') # REMOVE NOISE
                    full_text = clean_text.strip().lower()
                    
                    # Log Check: Is this the entity itself?
                    is_entity_self = False
                    for ent_idx in entity_indices:
                        if ent_idx >= current_word_start_idx and ent_idx < i:
                            is_entity_self = True
                            break
                    
                    if not is_entity_self and self.is_valid_word(full_text):
                        # Determine Direction
                        word_center = (current_word_start_idx + (i - 1)) / 2
                        direction = 'LEFT' if word_center < ent_center else 'RIGHT'
                        
                        words.append(TriggerInfo(
                            text=full_text,
                            score=current_word_score,
                            direction=direction,
                            tokens=current_word_parts
                        ))
                
                # Start New Word
                current_word_parts = [token]
                current_word_score = score
                current_word_start_idx = i
            else:
                # Append to current word
                current_word_parts.append(token)
                current_word_score += score

        # Flush last word
        if current_word_parts:
            # Clean up special tokens explicitly here
            raw_text = "".join(current_word_parts)
            clean_text = raw_text.replace('Ġ', '').replace(self.sp_marker, '').replace(' ', '')
            clean_text = clean_text.replace('</s>', '').replace('<s>', '') # REMOVE NOISE
            full_text = clean_text.strip().lower()
            
            is_entity_self = False
            for ent_idx in entity_indices:
                if ent_idx >= current_word_start_idx: is_entity_self = True
            
            if not is_entity_self and self.is_valid_word(full_text):
                word_center = (current_word_start_idx + len(tokens)) / 2
                direction = 'LEFT' if word_center < ent_center else 'RIGHT'
                words.append(TriggerInfo(text=full_text, score=current_word_score, direction=direction, tokens=current_word_parts))

        # 6. Sort by Score
        words.sort(key=lambda x: x.score, reverse=True)
        
        return words[:top_k]
