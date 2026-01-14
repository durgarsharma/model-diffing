#!/usr/bin/env python3
"""
Step 2.1: Fine-tune XNLI models - Create EN-tuned and HI-tuned variants
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Setup - optimized for Mac with Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f"Using device: {device}")

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create directories for saving models
os.makedirs("models/xnli_en_tuned", exist_ok=True)
os.makedirs("models/xnli_hi_tuned", exist_ok=True)

def load_and_prepare_nli_data(file_path, tokenizer, max_length=256):
    """Load and tokenize NLI data"""
    df = pd.read_csv(file_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['premise'],
            examples['hypothesis'],
            truncation=True,
            padding=False,  # We'll pad in the data collator
            max_length=max_length
        )
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['premise', 'hypothesis']
    )
    
    # Rename label column to labels (required by Trainer)
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def fine_tune_nli_model(train_data, eval_data, output_dir, language):
    """Fine-tune XLM-R for NLI task"""
    
    print(f"\n🚀 Fine-tuning XNLI model for {language}...")
    print(f"Training examples: {len(train_data)}")
    print(f"Evaluation examples: {len(eval_data)}")
    
    # Load fresh model for each fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )
    model.to(device)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments - optimized for Mac MPS memory constraints
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # Small number for quick training
        per_device_train_batch_size=4,  # Reduced from 8 for MPS memory
        per_device_eval_batch_size=8,   # Reduced from 16 for MPS memory
        gradient_accumulation_steps=2,   # Accumulate gradients to simulate larger batch
        warmup_steps=50,                # Reduced warmup steps
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,               # Less frequent logging
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb
        save_total_limit=1,  # Only keep 1 checkpoint to save memory
        dataloader_pin_memory=False,     # Disable pin memory for MPS
        fp16=False,                      # MPS doesn't support fp16
        bf16=False,                      # MPS doesn't support bf16
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print(f"Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate final model
    final_metrics = trainer.evaluate()
    print(f"Final metrics for {language}:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return trainer, final_metrics

# Main training process
if __name__ == "__main__":
    print("📚 Loading XNLI datasets...")
    
    # Load datasets
    en_train = load_and_prepare_nli_data("data/xnli/en_train.csv", tokenizer)
    hi_train = load_and_prepare_nli_data("data/xnli/hi_train.csv", tokenizer)
    en_test = load_and_prepare_nli_data("data/xnli/en_test.csv", tokenizer)
    hi_test = load_and_prepare_nli_data("data/xnli/hi_test.csv", tokenizer)
    
    print(f"✅ Data loaded successfully!")
    print(f"EN train: {len(en_train)}, EN test: {len(en_test)}")
    print(f"HI train: {len(hi_train)}, HI test: {len(hi_test)}")
    
    # Fine-tune English model
    print("\n" + "="*50)
    print("🇺🇸 FINE-TUNING ENGLISH MODEL")
    print("="*50)
    
    en_trainer, en_metrics = fine_tune_nli_model(
        train_data=en_train,
        eval_data=en_test,  # Use same language for validation during training
        output_dir="models/xnli_en_tuned",
        language="English"
    )
    
    # Fine-tune Hindi model
    print("\n" + "="*50)
    print("🇮🇳 FINE-TUNING HINDI MODEL")
    print("="*50)
    
    hi_trainer, hi_metrics = fine_tune_nli_model(
        train_data=hi_train,
        eval_data=hi_test,  # Use same language for validation during training
        output_dir="models/xnli_hi_tuned",
        language="Hindi"
    )
    
    # Save training summary
    summary = {
        'en_tuned_metrics': en_metrics,
        'hi_tuned_metrics': hi_metrics,
        'training_config': {
            'epochs': 2,
            'batch_size': 8,
            'model_name': model_name,
            'max_length': 256
        }
    }
    
    import json
    with open('results/xnli_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ XNLI Fine-tuning Complete!")
    print("Models saved:")
    print("  📁 models/xnli_en_tuned/")
    print("  📁 models/xnli_hi_tuned/")
    print("  📄 results/xnli_training_summary.json")
    
    print(f"\n📊 Quick Results Summary:")
    print(f"English-tuned model accuracy: {en_metrics['eval_accuracy']:.3f}")
    print(f"Hindi-tuned model accuracy: {hi_metrics['eval_accuracy']:.3f}")