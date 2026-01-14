#!/usr/bin/env python3
"""
Experiment 2: Confidence Analysis
Compare prediction confidence between EN-tuned and HI-tuned models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import os

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs("results/confidence_analysis", exist_ok=True)

def extract_prediction_confidence(model, inputs):
    """Extract prediction confidence from model outputs"""
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get prediction and confidence
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        max_confidence = torch.max(probabilities, dim=-1)[0].item()
        
        # Calculate entropy (uncertainty measure)
        prob_np = probabilities.cpu().numpy()[0]
        pred_entropy = entropy(prob_np + 1e-10)  # Add small epsilon
        
        return {
            'predicted_class': predicted_class,
            'max_confidence': max_confidence,
            'entropy': pred_entropy,
            'probabilities': prob_np
        }

def analyze_confidence_patterns():
    """Analyze confidence patterns across models and languages"""
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Load models
    en_model = AutoModelForSequenceClassification.from_pretrained("models/xnli_tuned/xnli_en_tuned").to(device)
    hi_model = AutoModelForSequenceClassification.from_pretrained("models/xnli_tuned/xnli_hi_tuned").to(device)
    
    # Load test data
    en_test = pd.read_csv("data/dataset/xnli/en_test.csv").head(200)  # Use subset for analysis
    hi_test = pd.read_csv("data/dataset/xnli/hi_test.csv").head(200)
    
    results = {
        'language': [],
        'model': [],
        'correct': [],
        'confidence': [],
        'entropy': [],
        'true_label': [],
        'predicted_label': []
    }
    
    label_names = ['entailment', 'neutral', 'contradiction']
    
    print("Analyzing confidence patterns...")
    
    # Process English test set
    print("Processing English test set...")
    for idx, row in en_test.iterrows():
        if idx % 50 == 0:
            print(f"  English sample {idx}/200")
            
        inputs = tokenizer(
            row['premise'], row['hypothesis'],
            return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(device)
        
        # Get predictions from both models
        for model_name, model in [('EN-tuned', en_model), ('HI-tuned', hi_model)]:
            pred_info = extract_prediction_confidence(model, inputs)
            
            results['language'].append('English')
            results['model'].append(model_name)
            results['correct'].append(pred_info['predicted_class'] == row['label'])
            results['confidence'].append(pred_info['max_confidence'])
            results['entropy'].append(pred_info['entropy'])
            results['true_label'].append(row['label'])
            results['predicted_label'].append(pred_info['predicted_class'])
    
    # Process Hindi test set
    print("Processing Hindi test set...")
    for idx, row in hi_test.iterrows():
        if idx % 50 == 0:
            print(f"  Hindi sample {idx}/200")
            
        inputs = tokenizer(
            row['premise'], row['hypothesis'],
            return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(device)
        
        # Get predictions from both models
        for model_name, model in [('EN-tuned', en_model), ('HI-tuned', hi_model)]:
            pred_info = extract_prediction_confidence(model, inputs)
            
            results['language'].append('Hindi')
            results['model'].append(model_name)
            results['correct'].append(pred_info['predicted_class'] == row['label'])
            results['confidence'].append(pred_info['max_confidence'])
            results['entropy'].append(pred_info['entropy'])
            results['true_label'].append(row['label'])
            results['predicted_label'].append(pred_info['predicted_class'])
    
    return pd.DataFrame(results)

def plot_1_confidence_by_correctness(df):
    """Plot 1: Confidence distribution by correctness"""
    plt.figure(figsize=(10, 6))
    
    correct_conf = df[df['correct'] == True]['confidence']
    incorrect_conf = df[df['correct'] == False]['confidence']
    
    plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct Predictions', 
             color='green', density=True)
    plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect Predictions', 
             color='red', density=True)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution by Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/confidence_analysis/1_confidence_by_correctness.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_2_confidence_by_model_language(df):
    """Plot 2: Confidence by model and language"""
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=df, x='language', y='confidence', hue='model')
    plt.title('Confidence by Model and Language')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/confidence_analysis/2_confidence_by_model_language.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_3_entropy_by_model_language(df):
    """Plot 3: Entropy by model and language"""
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=df, x='language', y='entropy', hue='model')
    plt.title('Prediction Entropy by Model and Language')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/confidence_analysis/3_entropy_by_model_language.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_4_confidence_calibration(df):
    """Plot 4: Confidence calibration scatter"""
    plt.figure(figsize=(12, 8))
    
    for model_name in ['EN-tuned', 'HI-tuned']:
        for lang in ['English', 'Hindi']:
            subset = df[(df['model'] == model_name) & (df['language'] == lang)]
            
            # Group by confidence bins and calculate accuracy
            conf_bins = np.linspace(0.33, 1.0, 10)
            bin_centers = []
            bin_accuracies = []
            
            for i in range(len(conf_bins) - 1):
                bin_mask = (subset['confidence'] >= conf_bins[i]) & (subset['confidence'] < conf_bins[i+1])
                if bin_mask.sum() > 0:
                    bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                    bin_accuracies.append(subset[bin_mask]['correct'].mean())
            
            if bin_centers:
                color = '#2E86AB' if model_name == 'EN-tuned' else '#A23B72'
                marker = 'o' if lang == 'English' else 's'
                plt.plot(bin_centers, bin_accuracies, marker=marker, color=color, 
                        alpha=0.7, label=f'{model_name} - {lang}', markersize=8, linewidth=2)
    
    plt.plot([0.33, 1.0], [0.33, 1.0], 'k--', alpha=0.5, label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Confidence Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/confidence_analysis/4_confidence_calibration.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_confidence_analysis(df):
    """Create all confidence analysis visualizations"""
    plot_1_confidence_by_correctness(df)
    plot_2_confidence_by_model_language(df) 
    plot_3_entropy_by_model_language(df)
    plot_4_confidence_calibration(df)

def analyze_confidence_by_correctness(df):
    """Analyze confidence patterns for correct vs incorrect predictions"""
    
    print("\nCONFIDENCE BY CORRECTNESS ANALYSIS")
    print("=" * 50)
    
    summary_stats = []
    
    for model_name in ['EN-tuned', 'HI-tuned']:
        for lang in ['English', 'Hindi']:
            subset = df[(df['model'] == model_name) & (df['language'] == lang)]
            
            correct_subset = subset[subset['correct'] == True]
            incorrect_subset = subset[subset['correct'] == False]
            
            correct_conf_mean = correct_subset['confidence'].mean()
            incorrect_conf_mean = incorrect_subset['confidence'].mean()
            
            correct_entropy_mean = correct_subset['entropy'].mean()
            incorrect_entropy_mean = incorrect_subset['entropy'].mean()
            
            print(f"\n{model_name} - {lang}:")
            print(f"  Correct predictions: {len(correct_subset)} samples")
            print(f"    Avg confidence: {correct_conf_mean:.3f}")
            print(f"    Avg entropy: {correct_entropy_mean:.3f}")
            print(f"  Incorrect predictions: {len(incorrect_subset)} samples")
            print(f"    Avg confidence: {incorrect_conf_mean:.3f}")
            print(f"    Avg entropy: {incorrect_entropy_mean:.3f}")
            print(f"  Confidence gap: {correct_conf_mean - incorrect_conf_mean:.3f}")
            
            summary_stats.append({
                'model': model_name,
                'language': lang,
                'correct_confidence': correct_conf_mean,
                'incorrect_confidence': incorrect_conf_mean,
                'confidence_gap': correct_conf_mean - incorrect_conf_mean,
                'correct_entropy': correct_entropy_mean,
                'incorrect_entropy': incorrect_entropy_mean
            })
    
    return summary_stats

def compute_calibration_metrics(df):
    """Compute confidence calibration metrics"""
    
    print("\nCALIBRATION ANALYSIS")
    print("=" * 30)
    
    calibration_results = {}
    
    for model_name in ['EN-tuned', 'HI-tuned']:
        for lang in ['English', 'Hindi']:
            subset = df[(df['model'] == model_name) & (df['language'] == lang)]
            
            # Expected Calibration Error (ECE)
            conf_bins = np.linspace(0, 1, 11)
            bin_boundaries = [(conf_bins[i], conf_bins[i+1]) for i in range(len(conf_bins)-1)]
            
            ece = 0
            total_samples = len(subset)
            
            for bin_lower, bin_upper in bin_boundaries:
                bin_mask = (subset['confidence'] > bin_lower) & (subset['confidence'] <= bin_upper)
                bin_data = subset[bin_mask]
                
                if len(bin_data) > 0:
                    bin_accuracy = bin_data['correct'].mean()
                    bin_confidence = bin_data['confidence'].mean()
                    bin_weight = len(bin_data) / total_samples
                    
                    ece += bin_weight * abs(bin_accuracy - bin_confidence)
            
            # Overconfidence measure
            avg_confidence = subset['confidence'].mean()
            avg_accuracy = subset['correct'].mean()
            overconfidence = avg_confidence - avg_accuracy
            
            print(f"{model_name} - {lang}:")
            print(f"  ECE: {ece:.3f}")
            print(f"  Avg Confidence: {avg_confidence:.3f}")
            print(f"  Avg Accuracy: {avg_accuracy:.3f}")
            print(f"  Overconfidence: {overconfidence:.3f}")
            
            calibration_results[f"{model_name}_{lang}"] = {
                'ece': ece,
                'overconfidence': overconfidence,
                'avg_confidence': avg_confidence,
                'avg_accuracy': avg_accuracy
            }
    
    return calibration_results

def main():
    """Run confidence analysis"""
    print("Starting Confidence Analysis...")
    print("=" * 60)
    
    # Analyze confidence patterns
    df = analyze_confidence_patterns()
    
    # Create visualizations
    plot_confidence_analysis(df)
    
    # Analyze confidence by correctness
    confidence_stats = analyze_confidence_by_correctness(df)
    
    # Compute calibration metrics
    calibration_results = compute_calibration_metrics(df)
    
    # Save results
    df.to_csv('results/confidence_analysis/confidence_data.csv', index=False)
    
    import json
    summary_results = {
        'confidence_stats': confidence_stats,
        'calibration_results': calibration_results
    }
    
    with open('results/confidence_analysis/confidence_results.json', 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    print(f"\nConfidence analysis complete!")
    print(f"Results saved to: results/confidence_analysis/")

if __name__ == "__main__":
    main()