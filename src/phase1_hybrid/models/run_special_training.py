
import os
import json
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from flexible_trainer import FlexibleNERTrainer
from crf_model import GreekLegalNERModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_conll(file_path):
    """Read CoNLL file and return list of sentences (words, labels)."""
    sentences = []
    current_words = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append({'tokens': current_words, 'ner_tags': current_labels})
                    current_words = []
                    current_labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    # Assuming last column is label, first is word. 
                    # Sometimes CoNLL has more columns.
                    word = parts[0]
                    label = parts[-1]
                    current_words.append(word)
                    current_labels.append(label)
    
    if current_words:
        sentences.append({'tokens': current_words, 'ner_tags': current_labels})
        
    return sentences

def convert_data(data_dir, output_dir):
    """Convert CoNLL files to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Map user files to standard names expected by loader or just custom names
    files = {
        'train': os.path.join(data_dir, 'testv2_conll.txt'), # User wants testv2 as train
        'dev': os.path.join(data_dir, 'validation_conll.txt')
    }
    
    json_files = {}
    
    for split, path in files.items():
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            continue
            
        logger.info(f"Converting {split} file: {path}")
        data = read_conll(path)
        
        # Save as JSON
        output_path = os.path.join(output_dir, f'{split}_special.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        json_files[split] = output_path
        logger.info(f"Saved to {output_path}. Sentences: {len(data)}")
        
    return json_files

class SpecialNERModel(GreekLegalNERModel):
    """Subclass to handle different tokenizers if needed."""
    def __init__(self, model_name, max_length=512, batch_size=16, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle tokenizer
        try:
            # Try with add_prefix_space=True (good for RoBERTa)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        except Exception:
            # Fallback for models that don't support it (like BERT/DeBERTa sometimes)
            logger.warning(f"Could not initialize tokenizer with add_prefix_space=True for {model_name}. Retrying without it.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        self.label_to_id = {}
        self.id_to_label = {}
        self.model = None
        
        print(f"🚀 Initialized SpecialNERModel")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")

def run_training(model_name, json_files, output_base_dir):
    """Run training for a specific model."""
    
    model_short_name = model_name.split('/')[-1]
    output_dir = os.path.join(output_base_dir, model_short_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting training for {model_name}...")
    
    # Initialize Data Model
    data_model = SpecialNERModel(
        model_name=model_name,
        max_length=512,
        batch_size=8 # Reduced batch size to be safe
    )
    
    # Load datasets
    # We use the same file for test as dev since user didn't provide a separate test set
    # or we can just ignore test. FlexibleTrainer expects test_dataloader though.
    train_path = json_files['train']
    dev_path = json_files['dev']
    
    # Load datasets
    # Note: load_datasets in GreekLegalNERModel expects 3 paths.
    # We will override load_datasets or just call the internal methods if possible.
    # GreekLegalNERModel.load_datasets calls _create_label_mappings and then creates datasets.
    
    # Let's just use the public method but pass dev path as test path too
    data_model.load_datasets(train_path, dev_path, dev_path)
    
    # Get dataloaders
    train_dataloader = data_model.train_loader
    val_dataloader = data_model.dev_loader
    test_dataloader = data_model.test_loader
    label_list = list(data_model.label_to_id.keys())
    
    # Initialize Model
    # We need to ensure the model is initialized with the correct number of labels
    # GreekLegalNERModel.load_datasets initializes self.model at the end
    model = data_model.model
    
    # Trainer
    trainer = FlexibleNERTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        label_list=label_list,
        output_dir=output_dir,
        learning_rate=2e-5,
        num_epochs=5, # Standard number of epochs
        save_steps=500,
        eval_steps=200
    )
    
    logger.info("Training...")
    trainer.train()
    
    logger.info(f"Training completed for {model_name}. Saved to {output_dir}")

def main():
    data_dir = r"C:\Users\User\Το Drive μου\AEGEAN UNIVERSITY\LEGAL DOCUMENTS ARCHIVE\ΠΑΙΓΑΙΟΥ\CODE\03_ML_MODELS\NER_MODELS\roberta_ner\datasets"
    output_base_dir = r"C:\Users\User\Το Drive μου\AEGEAN UNIVERSITY\LEGAL DOCUMENTS ARCHIVE\ΠΑΙΓΑΙΟΥ\CODE\03_ML_MODELS\NER_MODELS\roberta_ner\special_models_output"
    
    # 1. Convert Data
    json_files = convert_data(data_dir, data_dir) # Save JSONs in same dir
    
    if 'train' not in json_files or 'dev' not in json_files:
        logger.error("Missing data files. Aborting.")
        return

    # 2. Train Models
    models_to_train = [
        "AI-team-UoA/GreekLegalRoBERTa_v3",
        "AI-team-UoA/GreekDeBERTa-base"
    ]
    
    for model_name in models_to_train:
        try:
            run_training(model_name, json_files, output_base_dir)
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
