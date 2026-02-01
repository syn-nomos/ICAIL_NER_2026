import numpy as np
import torch
from typing import List, Dict, Optional

class AugmentedEmbeddingBuilder:
    """
    Constructs Augmented Vectors for the Active Learning Loop.
    Structure: [Prev_Token_Vector | Target_Vector | Next_Token_Vector | Type_One_Hot]
    
    Dimensions:
    - Context: 768 * 3 = 2304
    - Type: 9 (default)
    - Total: 2313
    """
    
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        
        # Taxonomy from the folder structure / schema
        self.types = sorted(['DATE', 'FACILITY', 'GPE', 'LEG_REFS', 'LOCATION', 'ORG', 'PERSON', 'PUBLIC_DOCS', 'O'])
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        self.type_dim = len(self.types)
        
    def get_type_vector(self, label: str) -> np.ndarray:
        """Returns the One-Hot vector for the given label."""
        vec = np.zeros(self.type_dim, dtype=np.float32)
        if label in self.type_to_idx:
            vec[self.type_to_idx[label]] = 1.0
        # If unknown label, return all zeros (or handle error?)
        return vec

    def _get_context_vector(self, embeddings: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Constructs the Context Signature: [Prev | Target | Next]
        embeddings: (seq_len, 768)
        start_idx: inclusive
        end_idx: exclusive
        """
        seq_len, dim = embeddings.shape
        safe_end = min(end_idx, seq_len)
        safe_start = min(start_idx, safe_end)
        
        # 1. Target (Mean of span)
        if safe_start == safe_end:
            target_vec = np.zeros(dim, dtype=np.float32)
        else:
            target_vec = np.mean(embeddings[safe_start:safe_end], axis=0)
            
        # 2. Prev (Immediate predecessor)
        if safe_start > 0:
            prev_vec = embeddings[safe_start - 1]
        else:
            prev_vec = np.zeros(dim, dtype=np.float32) # Zero padding for start of sentence
            
        # 3. Next (Immediate successor)
        if safe_end < seq_len:
            next_vec = embeddings[safe_end]
        else:
            next_vec = np.zeros(dim, dtype=np.float32) # Zero padding for end of sentence
            
        return np.concatenate([prev_vec, target_vec, next_vec])

    def build(self, roberta_wrapper, text: str, start_char: int, end_char: int, label: str = None) -> np.ndarray:
        """
        Main entry point to build an Augmented Vector for a specific entity span.
        
        Args:
            roberta_wrapper: Instance of RobertaNER (to access tokenizer/model).
            text: The full sentence text.
            start_char, end_char: Character offsets of the entity.
            label: The entity type (e.g. 'ORG'). If None, the One-Hot part will be all zeros.
            
        Returns:
            Augmented Vector (numpy array).
        """
        tokenizer = roberta_wrapper.tokenizer
        model = roberta_wrapper.model
        device = roberta_wrapper.device
        
        # 1. Tokenize & Embed
        # We need offsets to map char spans to token indices
        inputs = tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True, max_length=512)
        offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Get last hidden state (seq_len, 768)
        embeddings = outputs.hidden_states[-1][0].cpu().numpy()
        
        # 2. Find Token Indices
        # We find all tokens that overlap with the char span
        matching_indices = []
        for i, (o_start, o_end) in enumerate(offsets):
            if o_start == 0 and o_end == 0: continue # Skip special tokens
            
            # Intersection logic
            overlap_start = max(start_char, o_start)
            overlap_end = min(end_char, o_end)
            
            if overlap_end > overlap_start:
                matching_indices.append(i)
                
        if not matching_indices:
            # Fallback: Find closest token? Or return zeros?
            # Let's return zeros + label for safety, but log warning conceptually
            context_vec = np.zeros(self.embedding_dim * 3, dtype=np.float32)
        else:
            token_start = matching_indices[0]
            token_end = matching_indices[-1] + 1
            context_vec = self._get_context_vector(embeddings, token_start, token_end)
            
        # 3. Add Type Info
        type_vec = self.get_type_vector(label) if label else np.zeros(self.type_dim, dtype=np.float32)
        
        # 4. Concatenate
        augmented_vector = np.concatenate([context_vec, type_vec])
        
        return augmented_vector

    def batch_build(self, roberta_wrapper, text: str, entities: List[Dict]) -> List[np.ndarray]:
        """
        Builds vectors for multiple entities in the SAME sentence (optimization).
        entities: List of dicts {start, end, label}
        """
        tokenizer = roberta_wrapper.tokenizer
        model = roberta_wrapper.model
        device = roberta_wrapper.device
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True, max_length=512)
        offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        embeddings = outputs.hidden_states[-1][0].cpu().numpy()
        
        results = []
        for ent in entities:
            start_char, end_char = ent['start'], ent['end']
            label = ent.get('label')
            
            matching_indices = []
            for i, (o_start, o_end) in enumerate(offsets):
                if o_start == 0 and o_end == 0: continue
                overlap_start = max(start_char, o_start)
                overlap_end = min(end_char, o_end)
                if overlap_end > overlap_start:
                    matching_indices.append(i)
            
            if matching_indices:
                token_start = matching_indices[0]
                token_end = matching_indices[-1] + 1
                context_vec = self._get_context_vector(embeddings, token_start, token_end)
            else:
                context_vec = np.zeros(self.embedding_dim * 3, dtype=np.float32)
                
            type_vec = self.get_type_vector(label) if label else np.zeros(self.type_dim, dtype=np.float32)
            results.append(np.concatenate([context_vec, type_vec]))
            
        return results
