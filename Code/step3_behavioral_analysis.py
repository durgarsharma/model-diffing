#!/usr/bin/env python3
"""
Step 3: Behavioral Diffing Analysis for XNLI Models
Compare Base vs EN-tuned vs HI-tuned performance
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
os.makedirs("results/behavioral_analysis", exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def load_models():
    """Load all three models: Base, EN-tuned, HI-tuned"""
    print("Loading models...")
    
    # Base model - downloads from HuggingFace
    print("Downloading base XLM-RoBERTa model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", 
        num_labels=3
    ).to(device)
    
    # Fine-tuned models - load from your downloaded files
    print("Loading EN-tuned model...")
    en_tuned_model = AutoModelForSequenceClassification.from_pretrained(
        "models/xnli_tuned/xnli_en_tuned"  # Based on your file structure
    ).to(device)
    
    print("Loading HI-tuned model...")
    hi_tuned_model = AutoModelForSequenceClassification.from_pretrained(
        "models/xnli_tuned/xnli_hi_tuned"  # Based on your file structure
    ).to(device)
    
    return {
        'base': base_model,
        'en_tuned': en_tuned_model, 
        'hi_tuned': hi_tuned_model
    }

def load_test_data():
    """Load test datasets for both languages"""
    en_test = pd.read_csv("data/dataset/xnli/en_test.csv")
    hi_test = pd.read_csv("data/dataset/xnli/hi_test.csv")
    
    # Use subset for faster evaluation
    en_test = en_test.head(1000)
    hi_test = hi_test.head(1000)
    
    return {'en': en_test, 'hi': hi_test}

def evaluate_model(model, test_data, language):
    """Evaluate a model on test data"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            # Tokenize input
            inputs = tokenizer(
                row['premise'],
                row['hypothesis'],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            ).to(device)
            
            # Get prediction
            outputs = model(**inputs)
            predicted = torch.argmax(outputs.logits, dim=-1).item()
            
            predictions.append(predicted)
            true_labels.append(row['label'])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'language': language
    }

def run_behavioral_analysis():
    """Main behavioral analysis"""
    print("Starting Behavioral Analysis...")
    
    # Load models and data
    models = load_models()
    test_data = load_test_data()
    
    # Results storage
    results = {}
    
    # Evaluate all model-language combinations
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} model...")
        results[model_name] = {}
        
        for lang, data in test_data.items():
            print(f"  Testing on {lang.upper()}...")
            result = evaluate_model(model, data, lang)
            results[model_name][lang] = result
            print(f"  Accuracy: {result['accuracy']:.3f}")
    
    return results

def create_behavioral_diff_table(results):
    """Create delta comparison table"""
    # Extract accuracies
    base_en = results['base']['en']['accuracy']
    base_hi = results['base']['hi']['accuracy'] 
    en_tuned_en = results['en_tuned']['en']['accuracy']
    en_tuned_hi = results['en_tuned']['hi']['accuracy']
    hi_tuned_en = results['hi_tuned']['en']['accuracy']
    hi_tuned_hi = results['hi_tuned']['hi']['accuracy']
    
    # Calculate deltas vs base
    delta_data = {
        'Model': ['Base', 'EN-tuned', 'HI-tuned'],
        'English_Acc': [base_en, en_tuned_en, hi_tuned_en],
        'Hindi_Acc': [base_hi, en_tuned_hi, hi_tuned_hi],
        'Delta_EN': [0, en_tuned_en - base_en, hi_tuned_en - base_en],
        'Delta_HI': [0, en_tuned_hi - base_hi, hi_tuned_hi - base_hi],
        'Trade_off': [0, en_tuned_en - en_tuned_hi, hi_tuned_hi - hi_tuned_en]
    }
    
    df = pd.DataFrame(delta_data)
    return df

def plot_behavioral_differences(results, save_path="results/behavioral_analysis/"):
    """Create visualizations of behavioral differences"""
    
    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    models = ['Base', 'EN-tuned', 'HI-tuned']
    en_accs = [results['base']['en']['accuracy'], 
               results['en_tuned']['en']['accuracy'],
               results['hi_tuned']['en']['accuracy']]
    hi_accs = [results['base']['hi']['accuracy'],
               results['en_tuned']['hi']['accuracy'], 
               results['hi_tuned']['hi']['accuracy']]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.subplot(2, 2, 1)
    plt.bar(x - width/2, en_accs, width, label='English', alpha=0.8)
    plt.bar(x + width/2, hi_accs, width, label='Hindi', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Delta comparison
    plt.subplot(2, 2, 2)
    base_en, base_hi = results['base']['en']['accuracy'], results['base']['hi']['accuracy']
    
    delta_en = [0, 
                results['en_tuned']['en']['accuracy'] - base_en,
                results['hi_tuned']['en']['accuracy'] - base_en]
    delta_hi = [0,
                results['en_tuned']['hi']['accuracy'] - base_hi, 
                results['hi_tuned']['hi']['accuracy'] - base_hi]
    
    plt.bar(x - width/2, delta_en, width, label='Δ English', alpha=0.8)
    plt.bar(x + width/2, delta_hi, width, label='Δ Hindi', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('Accuracy Change vs Base')
    plt.title('Performance Changes vs Base Model')
    plt.xticks(x, models)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Trade-off visualization
    plt.subplot(2, 2, 3)
    trade_offs = [0,
                  results['en_tuned']['en']['accuracy'] - results['en_tuned']['hi']['accuracy'],
                  results['hi_tuned']['hi']['accuracy'] - results['hi_tuned']['en']['accuracy']]
    
    colors = ['gray', 'blue' if trade_offs[1] > 0 else 'red', 
              'orange' if trade_offs[2] > 0 else 'red']
    plt.bar(models, trade_offs, color=colors, alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('Language Preference (Acc_target - Acc_other)')
    plt.title('Language Trade-off Analysis')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # 4. Confusion matrices for EN-tuned model
    plt.subplot(2, 2, 4)
    cm_en = confusion_matrix(results['en_tuned']['en']['true_labels'], 
                            results['en_tuned']['en']['predictions'])
    sns.heatmap(cm_en, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Entail', 'Neutral', 'Contra'],
                yticklabels=['Entail', 'Neutral', 'Contra'])
    plt.title('EN-tuned Model: English Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}behavioral_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(results, delta_df):
    """Generate text summary of findings"""
    
    report = []
    report.append("BEHAVIORAL ANALYSIS SUMMARY")
    report.append("=" * 50)
    report.append("")
    
    # Model performances
    report.append("Model Performance:")
    for model_name in ['base', 'en_tuned', 'hi_tuned']:
        en_acc = results[model_name]['en']['accuracy']
        hi_acc = results[model_name]['hi']['accuracy']
        report.append(f"  {model_name.upper():10} | EN: {en_acc:.3f} | HI: {hi_acc:.3f}")
    
    report.append("")
    
    # Key findings
    base_en = results['base']['en']['accuracy']
    base_hi = results['base']['hi']['accuracy']
    en_tuned_en = results['en_tuned']['en']['accuracy']
    en_tuned_hi = results['en_tuned']['hi']['accuracy']
    hi_tuned_en = results['hi_tuned']['en']['accuracy']
    hi_tuned_hi = results['hi_tuned']['hi']['accuracy']
    
    report.append("Key Findings:")
    report.append(f"1. EN-tuning: EN {base_en:.3f}→{en_tuned_en:.3f} ({en_tuned_en-base_en:+.3f}), HI {base_hi:.3f}→{en_tuned_hi:.3f} ({en_tuned_hi-base_hi:+.3f})")
    report.append(f"2. HI-tuning: HI {base_hi:.3f}→{hi_tuned_hi:.3f} ({hi_tuned_hi-base_hi:+.3f}), EN {base_en:.3f}→{hi_tuned_en:.3f} ({hi_tuned_en-base_en:+.3f})")
    
    # Trade-off analysis
    en_trade_off = en_tuned_en - en_tuned_hi
    hi_trade_off = hi_tuned_hi - hi_tuned_en
    
    report.append("")
    report.append("Trade-off Analysis:")
    report.append(f"  EN-tuned preference: {en_trade_off:+.3f} (favors English)")
    report.append(f"  HI-tuned preference: {hi_trade_off:+.3f} (favors Hindi)")
    
    if en_tuned_hi < base_hi:
        report.append(f"  ⚠️ EN-tuning hurt Hindi by {base_hi - en_tuned_hi:.3f}")
    if hi_tuned_en < base_en:
        report.append(f"  ⚠️ HI-tuning hurt English by {base_en - hi_tuned_en:.3f}")
    
    return "\n".join(report)

def main():
    """Run complete behavioral analysis"""
    print("XNLI Behavioral Analysis Starting...")
    
    # Run analysis
    results = run_behavioral_analysis()
    
    # Create comparison table
    delta_df = create_behavioral_diff_table(results)
    print("\nDelta Comparison Table:")
    print(delta_df.to_string(index=False, float_format='%.3f'))
    
    # Save results
    delta_df.to_csv("results/behavioral_analysis/behavioral_diff_table.csv", index=False)
    
    # Create visualizations
    plot_behavioral_differences(results)
    
    # Generate summary report
    summary = generate_summary_report(results, delta_df)
    print(f"\n{summary}")
    
    # Save summary
    with open("results/behavioral_analysis/summary_report.txt", 'w') as f:
        f.write(summary)
    
    # Save raw results
    results_serializable = {}
    for model_name, model_results in results.items():
        results_serializable[model_name] = {}
        for lang, metrics in model_results.items():
            results_serializable[model_name][lang] = {
                'accuracy': float(metrics['accuracy']),
                'language': metrics['language']
            }
    
    with open("results/behavioral_analysis/raw_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\nBehavioral Analysis Complete!")
    print("Results saved in: results/behavioral_analysis/")

if __name__ == "__main__":
    main()
