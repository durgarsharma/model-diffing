#!/usr/bin/env python3
"""
Create 6 individual graphs for behavioral analysis results
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
os.makedirs("results/behavioral_analysis/individual_plots", exist_ok=True)

def evaluate_model_dynamic(model_path, test_csv, max_samples=500):
    """Dynamically evaluate model performance"""
    print(f"Evaluating {model_path} on {test_csv}...")
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    test_data = pd.read_csv(test_csv).head(max_samples)
    print(f"  Testing on {len(test_data)} samples")
    
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
    
    accuracy = correct / total
    print(f"  Accuracy: {accuracy:.3f}")
    return accuracy

def generate_results():
    """Generate results from model evaluation"""
    print("Generating results from model evaluation...")
    print("=" * 60)
    
    data = {
        'EN-tuned': {
            'English': evaluate_model_dynamic("models/xnli_tuned/xnli_en_tuned", "data/dataset/xnli/en_test.csv"),
            'Hindi': evaluate_model_dynamic("models/xnli_tuned/xnli_en_tuned", "data/dataset/xnli/hi_test.csv")
        },
        'HI-tuned': {
            'English': evaluate_model_dynamic("models/xnli_tuned/xnli_hi_tuned", "data/dataset/xnli/en_test.csv"),
            'Hindi': evaluate_model_dynamic("models/xnli_tuned/xnli_hi_tuned", "data/dataset/xnli/hi_test.csv")
        }
    }
    
    print("\nResults Generated:")
    print(f"EN-tuned: English {data['EN-tuned']['English']:.3f}, Hindi {data['EN-tuned']['Hindi']:.3f}")
    print(f"HI-tuned: English {data['HI-tuned']['English']:.3f}, Hindi {data['HI-tuned']['Hindi']:.3f}")
    
    return data

def plot_1_performance_comparison(data):
    """Plot 1: Performance Comparison Bar Chart"""
    plt.figure(figsize=(10, 6))
    
    models = ['EN-tuned', 'HI-tuned']
    en_scores = [data['EN-tuned']['English'], data['HI-tuned']['English']]
    hi_scores = [data['EN-tuned']['Hindi'], data['HI-tuned']['Hindi']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, en_scores, width, label='English', color='#2E86AB', alpha=0.8)
    bars2 = plt.bar(x + width/2, hi_scores, width, label='Hindi', color='#A23B72', alpha=0.8)
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance by Language', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(en_scores), max(hi_scores)) + 0.1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/behavioral_analysis/individual_plots/1_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_2_tradeoff_analysis(data):
    """Plot 2: Trade-off Analysis"""
    plt.figure(figsize=(10, 6))
    
    en_preference = data['EN-tuned']['English'] - data['EN-tuned']['Hindi']
    hi_preference = data['HI-tuned']['Hindi'] - data['HI-tuned']['English']
    
    models = ['EN-tuned', 'HI-tuned']
    trade_offs = [en_preference, hi_preference]
    colors = ['#2E86AB', '#A23B72']
    
    bars = plt.bar(models, trade_offs, color=colors, alpha=0.7)
    plt.ylabel('Language Preference (Target - Other)', fontsize=12)
    plt.title('Language Trade-off Analysis', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, trade_offs):
        plt.text(bar.get_x() + bar.get_width()/2., value + (0.005 if value > 0 else -0.005),
                f'{value:+.3f}', ha='center', va='bottom' if value > 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/behavioral_analysis/individual_plots/2_tradeoff_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_3_crosslang_impact(data):
    """Plot 3: Cross-language Impact"""
    plt.figure(figsize=(10, 6))
    
    hindi_impact = data['EN-tuned']['Hindi'] - data['HI-tuned']['Hindi']  # Negative = EN-tuning hurts Hindi
    english_impact = data['HI-tuned']['English'] - data['EN-tuned']['English']  # Negative = HI-tuning hurts English
    
    impacts = [hindi_impact, english_impact]
    impact_labels = ['Hindi\n(EN-tuning effect)', 'English\n(HI-tuning effect)']
    colors = ['#A23B72', '#2E86AB']
    
    bars = plt.bar(impact_labels, impacts, color=colors, alpha=0.7)
    plt.ylabel('Performance Impact', fontsize=12)
    plt.title('Cross-language Performance Impact', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, impacts):
        plt.text(bar.get_x() + bar.get_width()/2., 
                value + (0.005 if value > 0 else -0.005),
                f'{value:.3f}', ha='center', 
                va='bottom' if value > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/behavioral_analysis/individual_plots/3_crosslang_impact.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_4_heatmap(data):
    """Plot 4: Performance Heatmap"""
    plt.figure(figsize=(8, 6))
    
    heatmap_data = np.array([
        [data['EN-tuned']['English'], data['EN-tuned']['Hindi']],
        [data['HI-tuned']['English'], data['HI-tuned']['Hindi']]
    ])
    
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                     xticklabels=['English', 'Hindi'],
                     yticklabels=['EN-tuned', 'HI-tuned'],
                     cbar_kws={'label': 'Accuracy'})
    
    plt.title('Performance Heatmap', fontsize=14, fontweight='bold')
    plt.ylabel('Model', fontsize=12)
    plt.xlabel('Language', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/behavioral_analysis/individual_plots/4_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_5_specialization_scatter(data):
    """Plot 5: Language Specialization Scatter"""
    plt.figure(figsize=(10, 8))
    
    # Data points
    en_tuned_point = (data['EN-tuned']['Hindi'], data['EN-tuned']['English'])
    hi_tuned_point = (data['HI-tuned']['English'], data['HI-tuned']['Hindi'])
    
    plt.scatter(en_tuned_point[0], en_tuned_point[1], s=200, c='#2E86AB', alpha=0.7, label='EN-tuned')
    plt.scatter(hi_tuned_point[0], hi_tuned_point[1], s=200, c='#A23B72', alpha=0.7, label='HI-tuned')
    
    # Annotations
    plt.annotate('EN-tuned', en_tuned_point, xytext=(10, 10), textcoords='offset points')
    plt.annotate('HI-tuned', hi_tuned_point, xytext=(10, 10), textcoords='offset points')
    
    # Equal performance line
    min_val = min(min(en_tuned_point), min(hi_tuned_point)) - 0.02
    max_val = max(max(en_tuned_point), max(hi_tuned_point)) + 0.02
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equal Performance Line')
    
    plt.xlabel('Non-target Language Accuracy', fontsize=12)
    plt.ylabel('Target Language Accuracy', fontsize=12)
    plt.title('Target vs Non-target Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/behavioral_analysis/individual_plots/5_specialization_scatter.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_6_summary_table(data):
    """Plot 6: Summary Statistics Table"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate metrics
    en_preference = data['EN-tuned']['English'] - data['EN-tuned']['Hindi']
    hi_preference = data['HI-tuned']['Hindi'] - data['HI-tuned']['English']
    hindi_impact = data['EN-tuned']['Hindi'] - data['HI-tuned']['Hindi']
    english_impact = data['HI-tuned']['English'] - data['EN-tuned']['English']
    
    summary_data = [
        ['Metric', 'EN-tuned', 'HI-tuned'],
        ['English Accuracy', f"{data['EN-tuned']['English']:.3f}", f"{data['HI-tuned']['English']:.3f}"],
        ['Hindi Accuracy', f"{data['EN-tuned']['Hindi']:.3f}", f"{data['HI-tuned']['Hindi']:.3f}"],
        ['Language Preference', f"{en_preference:+.3f}", f"{hi_preference:+.3f}"],
        ['Cross-lang Impact', f"{hindi_impact:.3f}", f"{english_impact:.3f}"],
        ['Average Performance', f"{(data['EN-tuned']['English'] + data['EN-tuned']['Hindi'])/2:.3f}", 
         f"{(data['HI-tuned']['English'] + data['HI-tuned']['Hindi'])/2:.3f}"],
        ['Specialization Score', f"{abs(en_preference):.3f}", f"{abs(hi_preference):.3f}"]
    ]
    
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(3):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    ax.set_title('Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/behavioral_analysis/individual_plots/6_summary_table.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all individual plots"""
    print("Creating Individual Behavioral Analysis Plots...")
    print("=" * 60)
    
    # Generate dynamic results
    data = generate_results()
    
    # Create individual plots
    print("\nCreating individual plots...")
    plot_1_performance_comparison(data)
    plot_2_tradeoff_analysis(data)
    plot_3_crosslang_impact(data)
    plot_4_heatmap(data)
    plot_5_specialization_scatter(data)
    plot_6_summary_table(data)
    
    print("\nAll plots saved to: results/behavioral_analysis/individual_plots/")
    print("Files created:")
    print("  1_performance_comparison.png")
    print("  2_tradeoff_analysis.png")
    print("  3_crosslang_impact.png") 
    print("  4_heatmap.png")
    print("  5_specialization_scatter.png")
    print("  6_summary_table.png")

if __name__ == "__main__":
    main()