#!/usr/bin/env python3
"""
Debug Base Model Evaluation Issue
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def debug_base_model():
    """Debug why base model shows 33.3% accuracy"""
    
    print("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", 
        num_labels=3
    ).to(device)
    
    print("\n1. Model Architecture Check:")
    print(f"Model type: {type(base_model)}")
    print(f"Model config: {base_model.config}")
    print(f"Number of labels: {base_model.config.num_labels}")
    
    # Check classifier layer
    print(f"\nClassifier layer weights (first 5):")
    print(base_model.classifier.dense.weight[:5, :5])
    print(f"Classifier output projection shape: {base_model.classifier.out_proj.weight.shape}")
    
    print("\n2. Single Example Test:")
    premise = "The cat is sleeping on the couch."
    hypothesis_entail = "The cat is resting."  # Should be entailment
    hypothesis_contra = "The cat is running around."  # Should be contradiction
    hypothesis_neutral = "The couch is blue."  # Should be neutral
    
    test_cases = [
        (premise, hypothesis_entail, "entailment"),
        (premise, hypothesis_contra, "contradiction"), 
        (premise, hypothesis_neutral, "neutral")
    ]
    
    base_model.eval()
    with torch.no_grad():
        for prem, hyp, expected in test_cases:
            inputs = tokenizer(
                prem, hyp, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=256
            ).to(device)
            
            outputs = base_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted = torch.argmax(logits, dim=-1).item()
            
            label_names = ['entailment', 'neutral', 'contradiction']
            
            print(f"\nExample: {expected}")
            print(f"  Premise: {prem}")
            print(f"  Hypothesis: {hyp}")
            print(f"  Logits: {logits.cpu().numpy()}")
            print(f"  Probabilities: {probs.cpu().numpy()}")
            print(f"  Predicted: {label_names[predicted]} (index: {predicted})")
            print(f"  Confidence: {probs[0][predicted].item():.3f}")
    
    print("\n3. Batch Test on Real Data:")
    
    # Load test data
    try:
        en_test = pd.read_csv("data/xnli/en_test.csv").head(20)  # Small sample
        print(f"Loaded {len(en_test)} test examples")
        
        predictions = []
        true_labels = []
        confidences = []
        
        for _, row in en_test.iterrows():
            inputs = tokenizer(
                row['premise'], row['hypothesis'],
                return_tensors="pt",
                padding=True, 
                truncation=True,
                max_length=256
            ).to(device)
            
            outputs = base_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            conf = probs[0][pred].item()
            
            predictions.append(pred)
            true_labels.append(row['label'])
            confidences.append(conf)
        
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
        avg_confidence = np.mean(confidences)
        
        print(f"\nBatch Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Prediction distribution: {np.bincount(predictions)}")
        print(f"  True label distribution: {np.bincount(true_labels)}")
        
        # Show some examples
        print(f"\nFirst 5 examples:")
        for i in range(5):
            print(f"  True: {true_labels[i]}, Pred: {predictions[i]}, Conf: {confidences[i]:.3f}")
            
    except Exception as e:
        print(f"Error loading test data: {e}")
    
    print("\n4. Testing Different Model Loading Approaches:")
    
    # Try loading without specifying num_labels
    try:
        print("Loading without num_labels specification...")
        base_model_alt = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base"
        ).to(device)
        print(f"Alt model num_labels: {base_model_alt.config.num_labels}")
        
        # Test same example
        inputs = tokenizer(
            "The cat is sleeping.", "The cat is resting.",
            return_tensors="pt", padding=True, truncation=True
        ).to(device)
        
        outputs = base_model_alt(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        print(f"Alt model probabilities: {probs.cpu().numpy()}")
        
    except Exception as e:
        print(f"Error with alternative loading: {e}")

def compare_with_finetuned():
    """Compare base model behavior with fine-tuned models"""
    
    print("\n5. Comparing Base vs Fine-tuned Model Outputs:")
    
    # Load models
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=3
    ).to(device)
    
    en_tuned = AutoModelForSequenceClassification.from_pretrained(
        "models/xnli_en_tuned"
    ).to(device)
    
    # Test same input on both
    premise = "A person is reading a book."
    hypothesis = "Someone is reading."
    
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    with torch.no_grad():
        base_output = base_model(**inputs)
        en_output = en_tuned(**inputs)
        
        base_probs = torch.softmax(base_output.logits, dim=-1)
        en_probs = torch.softmax(en_output.logits, dim=-1)
        
        print(f"Base model probabilities: {base_probs.cpu().numpy()}")
        print(f"EN-tuned probabilities: {en_probs.cpu().numpy()}")
        print(f"Base prediction: {torch.argmax(base_probs, dim=-1).item()}")
        print(f"EN-tuned prediction: {torch.argmax(en_probs, dim=-1).item()}")

if __name__ == "__main__":
    print("DEBUGGING BASE MODEL EVALUATION")
    print("=" * 50)
    
    debug_base_model()
    compare_with_finetuned()
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS COMPLETE")
