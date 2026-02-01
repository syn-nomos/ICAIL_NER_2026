import numpy as np
import pickle
import os
from sklearn.linear_model import SGDClassifier
from typing import Dict, List, Optional

class BoundaryExpert:
    """
    A lightweight, online-learning model to validate entity boundaries.
    It uses Logistic Regression (SGD) to learn patterns in the Augmented Embeddings
    (specifically focusing on the Prev/Next context parts of the vector).
    
    It maintains a separate expert model for each Entity Type (ORG, GPE, etc.),
    as boundary patterns might differ by type.
    """
    
    def __init__(self, embedding_dim=2313):
        self.embedding_dim = embedding_dim
        # Dictionary mapping Label -> SGDClassifier
        self.experts: Dict[str, SGDClassifier] = {}
        self.classes = np.array([0, 1]) # 0: Bad Boundary, 1: Good Boundary

    def _get_expert(self, label: str) -> SGDClassifier:
        if label not in self.experts:
            # Initialize new expert with log loss (Logistic Regression)
            # penalty='l2' helps generalization
            # learning_rate='optimal' is good for SGD
            clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, random_state=42)
            self.experts[label] = clf
        return self.experts[label]

    def train_event(self, label: str, positive_vector: np.ndarray, negative_vectors: List[np.ndarray]):
        """
        updates the expert for 'label' with one positive example and multiple hard negatives.
        """
        clf = self._get_expert(label)
        
        # Prepare Batch
        # X: (1 + N_neg, dim)
        # y: (1 + N_neg,)
        X = [positive_vector] + negative_vectors
        y = [1] + [0] * len(negative_vectors)
        
        X = np.array(X)
        y = np.array(y)
        
        # Online Learning (partial_fit)
        # We iterate multiple times to ensure the model learns this specific example
        # (One-Shot Learning effect)
        for _ in range(10):
            clf.partial_fit(X, y, classes=self.classes)

    def predict_confidence(self, label: str, vector: np.ndarray) -> float:
        """
        Returns the probability (0.0 to 1.0) that this vector represents a Valid Boundary
        for the given label.
        """
        if label not in self.experts:
            # If we know nothing about this label, return 0.5 (uncertainty)
            # or maybe slightly lower to be conservative?
            return 0.5
            
        clf = self.experts[label]
        
        # Check if the model is fitted (has learned at least once)
        try:
            # predict_proba returns [[prob_0, prob_1]]
            cutoff = clf.predict_proba([vector])[0][1]
            return float(cutoff)
        except Exception:
            # Fallback if not fully initialized (e.g. sklearn version quirks)
            return 0.5

    def save(self, directory: str):
        """Saves the experts to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        params = {
            'experts': self.experts,
            'dim': self.embedding_dim
        }
        with open(os.path.join(directory, 'boundary_experts.pkl'), 'wb') as f:
            pickle.dump(params, f)

    def load(self, directory: str):
        """Loads experts from disk."""
        path = os.path.join(directory, 'boundary_experts.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                params = pickle.load(f)
                self.experts = params['experts']
                self.embedding_dim = params['dim']
            print(f"✅ Loaded {len(self.experts)} Boundary Experts.")
        else:
            print("⚠️ No saved experts found. Starting fresh.")
