#!/usr/bin/env python3
"""
Experiment 3: Error Analysis
Analyze error patterns between EN-tuned and HI-tuned models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs("results/error_analysis", exist_ok=True)

def get_model_predictions(model, test_data, tokenizer, max_samples=300):
    """Get predictions from a model on test data"""
    model.eval()
    
    predictions = []
    true_labels = []
    confidences = []
    
    for idx, row in test_data.head(max_samples).iterrows():
        if idx % 50 == 0:
            print(f"  Processing sample {idx}/{max_samples}")
            
        inputs = tokenizer(
            row['premise'], row['hypothesis'],
            return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1)[0].item()
        
        predictions.append(predicted)
        true_labels.append(row['label'])
        confidences.append(confidence)
    
    return predictions, true_labels, confidences

def analyze_error_patterns():
    """Analyze error patterns across models and languages"""
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Load models
    en_model = AutoModelForSequenceClassification.from_pretrained("models/xnli_tuned/xnli_en_tuned").to(device)
    hi_model = AutoModelForSequenceClassification.from_pretrained("models/xnli_tuned/xnli_hi_tuned").to(device)
    
    # Load test data
    en_test = pd.read_csv("data/dataset/xnli/en_test.csv")
    hi_test = pd.read_csv("data/dataset/xnli/hi_test.csv")
    
    results = {}
    
    print("Analyzing error patterns...")
    
    # Get predictions for all model-language combinations
    print("EN-tuned model on English...")
    en_model_en_preds, en_model_en_true, en_model_en_conf = get_model_predictions(en_model, en_test, tokenizer)
    
    print("EN-tuned model on Hindi...")
    en_model_hi_preds, en_model_hi_true, en_model_hi_conf = get_model_predictions(en_model, hi_test, tokenizer)
    
    print("HI-tuned model on English...")
    hi_model_en_preds, hi_model_en_true, hi_model_en_conf = get_model_predictions(hi_model, en_test, tokenizer)
    
    print("HI-tuned model on Hindi...")
    hi_model_hi_preds, hi_model_hi_true, hi_model_hi_conf = get_model_predictions(hi_model, hi_test, tokenizer)
    
    results = {
        'EN_model_EN_data': {'pred': en_model_en_preds, 'true': en_model_en_true, 'conf': en_model_en_conf},
        'EN_model_HI_data': {'pred': en_model_hi_preds, 'true': en_model_hi_true, 'conf': en_model_hi_conf},
        'HI_model_EN_data': {'pred': hi_model_en_preds, 'true': hi_model_en_true, 'conf': hi_model_en_conf},
        'HI_model_HI_data': {'pred': hi_model_hi_preds, 'true': hi_model_hi_true, 'conf': hi_model_hi_conf}
    }
    
    return results

def plot_1_confusion_matrices(results):
    """Plot 1: Confusion matrices for all model-language combinations"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    label_names = ['Entailment', 'Neutral', 'Contradiction']
    
    # EN-tuned on English
    cm1 = confusion_matrix(results['EN_model_EN_data']['true'], results['EN_model_EN_data']['pred'])
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=label_names, yticklabels=label_names)
    ax1.set_title('EN-tuned on English')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # EN-tuned on Hindi
    cm2 = confusion_matrix(results['EN_model_HI_data']['true'], results['EN_model_HI_data']['pred'])
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=label_names, yticklabels=label_names)
    ax2.set_title('EN-tuned on Hindi')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # HI-tuned on English
    cm3 = confusion_matrix(results['HI_model_EN_data']['true'], results['HI_model_EN_data']['pred'])
    sns.heatmap(cm3, annot=True, fmt='d', cmap='Greens', ax=ax3,
                xticklabels=label_names, yticklabels=label_names)
    ax3.set_title('HI-tuned on English')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # HI-tuned on Hindi
    cm4 = confusion_matrix(results['HI_model_HI_data']['true'], results['HI_model_HI_data']['pred'])
    sns.heatmap(cm4, annot=True, fmt='d', cmap='Purples', ax=ax4,
                xticklabels=label_names, yticklabels=label_names)
    ax4.set_title('HI-tuned on Hindi')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('results/error_analysis/1_confusion_matrices.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_2_accuracy_by_label(results):
    """Plot 2: Accuracy by NLI label category"""
    plt.figure(figsize=(12, 6))
    
    label_names = ['Entailment', 'Neutral', 'Contradiction']
    model_lang_combinations = [
        ('EN_model_EN_data', 'EN-tuned on English', '#2E86AB'),
        ('EN_model_HI_data', 'EN-tuned on Hindi', '#87CEEB'),
        ('HI_model_EN_data', 'HI-tuned on English', '#DDA0DD'),
        ('HI_model_HI_data', 'HI-tuned on Hindi', '#A23B72')
    ]
    
    x = np.arange(len(label_names))
    width = 0.2
    
    for i, (key, label, color) in enumerate(model_lang_combinations):
        true_labels = np.array(results[key]['true'])
        pred_labels = np.array(results[key]['pred'])
        
        accuracies = []
        for label_idx in range(3):
            mask = (true_labels == label_idx)
            if mask.sum() > 0:
                accuracy = (pred_labels[mask] == label_idx).mean()
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        plt.bar(x + i * width, accuracies, width, label=label, color=color, alpha=0.8)
    
    plt.xlabel('NLI Label')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by NLI Label Category')
    plt.xticks(x + width * 1.5, label_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/error_analysis/2_accuracy_by_label.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_3_error_type_analysis(results):
    """Plot 3: Error type distribution"""
    plt.figure(figsize=(12, 8))
    
    label_names = ['Entailment', 'Neutral', 'Contradiction']
    
    error_data = []
    
    for key, name in [('EN_model_EN_data', 'EN→EN'), ('EN_model_HI_data', 'EN→HI'),
                      ('HI_model_EN_data', 'HI→EN'), ('HI_model_HI_data', 'HI→HI')]:
        
        true_labels = np.array(results[key]['true'])
        pred_labels = np.array(results[key]['pred'])
        
        # Count error types
        error_types = {
            'Entailment→Neutral': 0, 'Entailment→Contradiction': 0,
            'Neutral→Entailment': 0, 'Neutral→Contradiction': 0,
            'Contradiction→Entailment': 0, 'Contradiction→Neutral': 0
        }
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label != pred_label:
                true_name = label_names[true_label]
                pred_name = label_names[pred_label]
                error_type = f"{true_name}→{pred_name}"
                if error_type in error_types:
                    error_types[error_type] += 1
        
        # Convert to percentages
        total_errors = sum(error_types.values())
        if total_errors > 0:
            for error_type in error_types:
                error_types[error_type] = error_types[error_type] / total_errors * 100
        
        for error_type, percentage in error_types.items():
            error_data.append({
                'Model_Data': name,
                'Error_Type': error_type,
                'Percentage': percentage
            })
    
    df = pd.DataFrame(error_data)
    
    # Create grouped bar plot
    error_types_list = list(set(df['Error_Type']))
    model_combinations = ['EN→EN', 'EN→HI', 'HI→EN', 'HI→HI']
    
    x = np.arange(len(error_types_list))
    width = 0.2
    colors = ['#2E86AB', '#87CEEB', '#DDA0DD', '#A23B72']
    
    for i, (model_combo, color) in enumerate(zip(model_combinations, colors)):
        subset = df[df['Model_Data'] == model_combo]
        percentages = []
        for error_type in error_types_list:
            matching = subset[subset['Error_Type'] == error_type]
            if len(matching) > 0:
                percentages.append(matching['Percentage'].iloc[0])
            else:
                percentages.append(0)
        
        plt.bar(x + i * width, percentages, width, label=model_combo, color=color, alpha=0.8)
    
    plt.xlabel('Error Type')
    plt.ylabel('Percentage of Total Errors')
    plt.title('Error Type Distribution')
    plt.xticks(x + width * 1.5, error_types_list, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/error_analysis/3_error_type_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_4_model_comparison_radar(results):
    """Plot 4: Radar chart comparing model performance"""
    from math import pi
    
    # Calculate metrics for each model-language combination
    metrics = {}
    label_names = ['Entailment', 'Neutral', 'Contradiction']
    
    for key, display_name in [('EN_model_EN_data', 'EN-tuned on English'), 
                             ('HI_model_HI_data', 'HI-tuned on Hindi')]:
        
        true_labels = np.array(results[key]['true'])
        pred_labels = np.array(results[key]['pred'])
        
        # Calculate per-class accuracies
        class_accuracies = []
        for label_idx in range(3):
            mask = (true_labels == label_idx)
            if mask.sum() > 0:
                accuracy = (pred_labels[mask] == label_idx).mean()
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)
        
        overall_accuracy = (pred_labels == true_labels).mean()
        
        metrics[display_name] = {
            'Overall': overall_accuracy,
            'Entailment': class_accuracies[0],
            'Neutral': class_accuracies[1], 
            'Contradiction': class_accuracies[2]
        }
    
    # Create radar chart
    categories = list(metrics['EN-tuned on English'].keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#2E86AB', '#A23B72']
    for i, (model_name, color) in enumerate(zip(metrics.keys(), colors)):
        values = list(metrics[model_name].values())
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison (Target Languages)', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/error_analysis/4_model_comparison_radar.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_error_analysis_summary(results):
    """Generate summary of error analysis findings"""
    
    print("\nERROR ANALYSIS SUMMARY")
    print("=" * 50)
    
    label_names = ['Entailment', 'Neutral', 'Contradiction']
    
    # Calculate overall accuracies
    for key, display_name in [('EN_model_EN_data', 'EN-tuned on English'),
                             ('EN_model_HI_data', 'EN-tuned on Hindi'),
                             ('HI_model_EN_data', 'HI-tuned on English'),
                             ('HI_model_HI_data', 'HI-tuned on Hindi')]:
        
        true_labels = np.array(results[key]['true'])
        pred_labels = np.array(results[key]['pred'])
        
        overall_accuracy = (pred_labels == true_labels).mean()
        
        # Per-class accuracies
        class_accuracies = []
        for label_idx in range(3):
            mask = (true_labels == label_idx)
            if mask.sum() > 0:
                accuracy = (pred_labels[mask] == label_idx).mean()
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)
        
        print(f"\n{display_name}:")
        print(f"  Overall Accuracy: {overall_accuracy:.3f}")
        for i, (label, acc) in enumerate(zip(label_names, class_accuracies)):
            print(f"  {label} Accuracy: {acc:.3f}")
    
    # Most confused pairs
    print(f"\nMost Common Error Patterns:")
    
    all_confusion_pairs = {}
    
    for key in results.keys():
        true_labels = np.array(results[key]['true'])
        pred_labels = np.array(results[key]['pred'])
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label != pred_label:
                pair = f"{label_names[true_label]}→{label_names[pred_label]}"
                all_confusion_pairs[pair] = all_confusion_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(all_confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    for pair, count in sorted_pairs[:5]:
        print(f"  {pair}: {count} errors")

def main():
    """Run error analysis"""
    print("Starting Error Analysis...")
    print("=" * 60)
    
    # Analyze error patterns
    results = analyze_error_patterns()
    
    # Create individual plots
    plot_1_confusion_matrices(results)
    plot_2_accuracy_by_label(results)
    plot_3_error_type_analysis(results)
    plot_4_model_comparison_radar(results)
    
    # Generate summary
    generate_error_analysis_summary(results)
    
    # Save results
    import json
    summary_data = {}
    
    for key in results.keys():
        true_labels = results[key]['true']
        pred_labels = results[key]['pred']
        accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
        summary_data[key] = {'accuracy': accuracy}
    
    with open('results/error_analysis/error_results.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nError analysis complete!")
    print(f"Results saved to: results/error_analysis/")
    print(f"Individual plots created:")
    print(f"  1_confusion_matrices.png")
    print(f"  2_accuracy_by_label.png") 
    print(f"  3_error_type_analysis.png")
    print(f"  4_model_comparison_radar.png")

if __name__ == "__main__":
    main()
