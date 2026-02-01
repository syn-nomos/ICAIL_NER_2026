
import os
import logging
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    AutoConfig
)
from seqeval.metrics import classification_report

import sys
import argparse

# --- CONFIGURATION ---
# Default values
DEFAULT_MODEL_NAME = "src/agents/GreekLegalRoBERTa_v3"
DEFAULT_TRAIN_FILE = "dataset/export_conll_2026_01_14-revised.txt"
DEFAULT_OUTPUT_DIR = "src/agents/GreekLegalRoBERTa_New"
MAX_LEN = 128

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train RoBERTa NER")
    parser.add_argument("--input_file", type=str, default=DEFAULT_TRAIN_FILE, help="Path to CoNLL training file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL_NAME, help="Base model path or name")
    return parser.parse_args()

def read_conll(file_path):
    """Reads a CoNLL file and returns a list of sentences with tokens and tags."""
    logger.info(f"Reading CoNLL file: {file_path}")
    sentences = []
    tokens = []
    tags = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens = []
                    tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1]) # Assuming last column is the tag
                    
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
        
    logger.info(f"Found {len(sentences)} sentences.")
    return sentences

def get_label_list(sentences):
    """Extracts unique labels from the dataset."""
    unique_labels = set()
    for sent in sentences:
        unique_labels.update(sent["ner_tags"])
    label_list = sorted(list(unique_labels))
    return label_list

def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        max_length=MAX_LEN
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    args = parse_args()
    
    # 1. Prepare Data
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return

    raw_data = read_conll(args.input_file)
    if not raw_data:
        logger.error("No data found in input file.")
        return

    # --- LABEL DETECTION STRATEGY ---
    # Prioritize Existing Model's Labels to prevent ID mismatch or forgetting
    labels_from_data = get_label_list(raw_data)
    
    label_list = []
    label_to_id = {}
    id_to_label = {}
    
    # Try to load strict config from base model
    try:
        config = AutoConfig.from_pretrained(args.base_model)
        if config.id2label:
            logger.info(f"♻️  Inheriting {len(config.id2label)} labels from Base Model config to preserve ID alignment.")
            id_to_label = {int(k): v for k, v in config.id2label.items()}
            label_to_id = {v: int(k) for k, v in config.id2label.items()}
            label_list = list(label_to_id.keys())
            
            # Check for new labels in data that weren't in model
            new_labels = [l for l in labels_from_data if l not in label_list]
            if new_labels:
                logger.warning(f"⚠️  Found NEW labels in data not in model: {new_labels}. Extending head...")
                # Add them
                start_id = max(id_to_label.keys()) + 1
                for nl in new_labels:
                    id_to_label[start_id] = nl
                    label_to_id[nl] = start_id
                    label_list.append(nl)
                    start_id += 1
        else:
            raise ValueError("No id2label in config")
            
    except Exception as e:
        logger.info(f"🆕 Creating fresh label map (No existing config found or generic): {e}")
        label_list = sorted(list(labels_from_data))
        label_to_id = {l: i for i, l in enumerate(label_list)}
        id_to_label = {i: l for i, l in enumerate(label_list)}

    logger.info(f"Final Label Count: {len(label_list)}")
    
    # Create HF Dataset
    full_dataset = Dataset.from_list(raw_data)
    # If dataset is small, maybe use all for train? But we need validation.
    if len(full_dataset) < 10:
        logger.warning("Dataset too small, using all for training (no validation).")
        datasets = DatasetDict({"train": full_dataset, "validation": full_dataset})
    else:
        dataset_split = full_dataset.train_test_split(test_size=0.1)
        datasets = DatasetDict({
            "train": dataset_split["train"],
            "validation": dataset_split["test"]
        })

    # 2. Tokenizer & Model
    logger.info(f"Loading tokenizer: {args.base_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", add_prefix_space=True)
    
    logger.info(f"Loading model: {args.base_model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True # Allow resizing head if labels changed
    )

    # 3. Tokenization
    tokenized_datasets = datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",  # Updated from evaluation_strategy for newer transformers
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=2,
        overwrite_output_dir=True 
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.train()
    
    # 7. Save Final Model
    logger.info(f"Saving model to {args.output_dir}")
    try:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Training Complete!")
    except Exception as e:
        logger.error(f"❌ Could not save to {args.output_dir}: {e}")
        # Fallback to timestamped directory to save work
        import time
        timestamp = int(time.time())
        fallback_dir = f"{args.output_dir}_{timestamp}"
        logger.info(f"⚠️  Attempting fallback save to: {fallback_dir}")
        try:
            trainer.save_model(fallback_dir)
            tokenizer.save_pretrained(fallback_dir)
            logger.info(f"✅ Saved successfully to fallback: {fallback_dir}")
        except Exception as e2:
            logger.error(f"❌ Fallback also failed: {e2}")

if __name__ == "__main__":
    main()
