#!/usr/bin/env python3
"""
Corrected Behavioral Analysis - Compare Fine-tuned Models Only
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def evaluate_model_corrected(model_path, test_csv, max_samples=500):
    """Evaluate model with proper local path loading"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    test_data = pd.read_csv(test_csv).head(max_samples)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            inputs = tokenizer(
                row['premise'], row['hypothesis'],
                return_tensors="pt", truncation=True, padding=True, max_length=256
            ).to(device)
            
            outputs = model(**inputs)
            predicted = torch.argmax(outputs.logits, dim=-1).item()
            
            if predicted == row['label']:
                correct += 1
            total += 1
    
    return correct / total

# Evaluate both models on both languages
print("Corrected Analysis: Fine-tuned Models Comparison")
print("=" * 60)

en_model_en = evaluate_model_corrected("models/xnli_tuned/xnli_en_tuned", "data/dataset/xnli/en_test.csv")
en_model_hi = evaluate_model_corrected("models/xnli_tuned/xnli_en_tuned", "data/dataset/xnli/hi_test.csv") 
hi_model_en = evaluate_model_corrected("models/xnli_tuned/xnli_hi_tuned", "data/dataset/xnli/en_test.csv")
hi_model_hi = evaluate_model_corrected("models/xnli_tuned/xnli_hi_tuned", "data/dataset/xnli/hi_test.csv")

print(f"\nResults:")
print(f"EN-tuned model: English {en_model_en:.3f}, Hindi {en_model_hi:.3f}")
print(f"HI-tuned model: English {hi_model_en:.3f}, Hindi {hi_model_hi:.3f}")

print(f"\nTrade-off Analysis:")
en_preference = en_model_en - en_model_hi
hi_preference = hi_model_hi - hi_model_en

print(f"EN-tuned language preference: {en_preference:+.3f} (favors English)")
print(f"HI-tuned language preference: {hi_preference:+.3f} (favors Hindi)")

if en_preference > 0:
    print(f"✓ EN-tuning created English advantage")
if hi_preference > 0:
    print(f"✓ HI-tuning created Hindi advantage")

# Cross-language impact
print(f"\nCross-language Impact:")
if en_model_hi < hi_model_hi:
    print(f"EN-tuning reduced Hindi performance by {hi_model_hi - en_model_hi:.3f}")
if hi_model_en < en_model_en:
    print(f"HI-tuning reduced English performance by {en_model_en - hi_model_en:.3f}")
