#!/usr/bin/env python3
"""
Experiment 1: Attention Pattern Analysis
Compare attention patterns between EN-tuned and HI-tuned models
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

os.makedirs("results/attention_analysis", exist_ok=True)

def extract_attention_patterns(model, inputs, tokenizer):
    """Extract attention weights from all layers and heads"""
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # Tuple of (batch_size, num_heads, seq_len, seq_len)
        
        # Convert to numpy and aggregate
        attention_patterns = []
        for layer_idx, layer_attention in enumerate(attentions):
            # Average across heads for simplicity
            avg_attention = layer_attention.mean(dim=1).cpu().numpy()  # (batch_size, seq_len, seq_len)
            attention_patterns.append(avg_attention[0])  # Take first (only) sample
    
    return attention_patterns

def compute_attention_entropy(attention_matrix):
    """Compute entropy of attention distribution"""
    # For each query token, compute entropy of its attention distribution
    entropies = []
    for i in range(attention_matrix.shape[0]):
        attention_dist = attention_matrix[i]
        # Add small epsilon to avoid log(0)
        attention_dist = attention_dist + 1e-10
        attention_dist = attention_dist / attention_dist.sum()  # Normalize
        ent = entropy(attention_dist)
        entropies.append(ent)
    
    return np.mean(entropies)

def analyze_attention_patterns():
    """Compare attention patterns between models"""
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Load models
    en_model = AutoModelForSequenceClassification.from_pretrained("models/xnli_tuned/xnli_en_tuned").to(device)
    hi_model = AutoModelForSequenceClassification.from_pretrained("models/xnli_tuned/xnli_hi_tuned").to(device)
    
    # Create test examples
    test_examples = [
        ("A man is reading a book.", "Someone is reading.", "English"),
        ("The cat is sleeping.", "The animal is resting.", "English"), 
        ("एक आदमी किताब पढ़ रहा है।", "कोई व्यक्ति पढ़ रहा है।", "Hindi"),
        ("बिल्ली सो रही है।", "जानवर आराम कर रहा है।", "Hindi")
    ]
    
    results = {
        'examples': [],
        'en_model_entropy': [],
        'hi_model_entropy': [],
        'en_model_attention': [],
        'hi_model_attention': [],
        'language': []
    }
    
    print("Analyzing attention patterns...")
    
    for premise, hypothesis, lang in test_examples:
        print(f"Processing {lang} example...")
        
        # Tokenize
        inputs = tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        
        # Get token strings for visualization
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Extract attention patterns
        en_attention = extract_attention_patterns(en_model, inputs, tokenizer)
        hi_attention = extract_attention_patterns(hi_model, inputs, tokenizer)
        
        # Compute attention entropy for each layer
        en_entropies = [compute_attention_entropy(att) for att in en_attention]
        hi_entropies = [compute_attention_entropy(att) for att in hi_attention]
        
        # Store results
        results['examples'].append(f"{premise[:30]}...")
        results['en_model_entropy'].append(np.mean(en_entropies))
        results['hi_model_entropy'].append(np.mean(hi_entropies))
        results['en_model_attention'].append(en_attention)
        results['hi_model_attention'].append(hi_attention)
        results['language'].append(lang)
        
        print(f"  EN-model avg entropy: {np.mean(en_entropies):.3f}")
        print(f"  HI-model avg entropy: {np.mean(hi_entropies):.3f}")
    
    return results, test_examples

def plot_attention_entropy_comparison(results):
    """Plot attention entropy comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Entropy by example
    examples = results['examples']
    en_entropy = results['en_model_entropy']
    hi_entropy = results['hi_model_entropy']
    languages = results['language']
    
    x = np.arange(len(examples))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, en_entropy, width, label='EN-tuned Model', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, hi_entropy, width, label='HI-tuned Model', 
                   color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Test Examples')
    ax1.set_ylabel('Average Attention Entropy')
    ax1.set_title('Attention Entropy by Model and Example')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{ex}\n({lang})" for ex, lang in zip(examples, languages)], 
                       rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Entropy by language
    en_examples = [i for i, lang in enumerate(languages) if lang == 'English']
    hi_examples = [i for i, lang in enumerate(languages) if lang == 'Hindi']
    
    en_lang_en_model = np.mean([en_entropy[i] for i in en_examples])
    en_lang_hi_model = np.mean([hi_entropy[i] for i in en_examples])
    hi_lang_en_model = np.mean([en_entropy[i] for i in hi_examples])
    hi_lang_hi_model = np.mean([hi_entropy[i] for i in hi_examples])
    
    lang_data = {
        'Language': ['English', 'English', 'Hindi', 'Hindi'],
        'Model': ['EN-tuned', 'HI-tuned', 'EN-tuned', 'HI-tuned'],
        'Entropy': [en_lang_en_model, en_lang_hi_model, hi_lang_en_model, hi_lang_hi_model]
    }
    
    df = pd.DataFrame(lang_data)
    
    # Pivot for grouped bar chart
    pivot_df = df.pivot(index='Language', columns='Model', values='Entropy')
    
    pivot_df.plot(kind='bar', ax=ax2, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax2.set_ylabel('Average Attention Entropy')
    ax2.set_title('Attention Entropy by Language and Model')
    ax2.legend(title='Model')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=0)
    
    # Add value labels
    for i, lang in enumerate(['English', 'Hindi']):
        en_val = pivot_df.loc[lang, 'EN-tuned']
        hi_val = pivot_df.loc[lang, 'HI-tuned']
        ax2.text(i - 0.2, en_val + 0.01, f'{en_val:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + 0.2, hi_val + 0.01, f'{hi_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/attention_analysis/attention_entropy_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_attention_heatmap(results, test_examples, example_idx=0):
    """Plot attention heatmap for a specific example"""
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    premise, hypothesis, lang = test_examples[example_idx]
    
    # Get tokens for the example
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Get attention patterns for this example
    en_attention = results['en_model_attention'][example_idx]
    hi_attention = results['hi_model_attention'][example_idx]
    
    # Plot attention for the last layer (most task-relevant)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Trim to actual sequence length (remove padding)
    seq_len = len([t for t in tokens if t != '<pad>'])
    display_tokens = tokens[:seq_len]
    
    # EN-tuned model attention (last layer)
    en_att_last = en_attention[-1][:seq_len, :seq_len]
    sns.heatmap(en_att_last, 
               xticklabels=display_tokens, 
               yticklabels=display_tokens,
               cmap='Blues', ax=ax1, cbar_kws={'label': 'Attention Weight'})
    ax1.set_title(f'EN-tuned Model Attention\n({lang} Example)')
    ax1.set_xlabel('Key Tokens')
    ax1.set_ylabel('Query Tokens')
    
    # HI-tuned model attention (last layer)
    hi_att_last = hi_attention[-1][:seq_len, :seq_len]
    sns.heatmap(hi_att_last,
               xticklabels=display_tokens,
               yticklabels=display_tokens, 
               cmap='Reds', ax=ax2, cbar_kws={'label': 'Attention Weight'})
    ax2.set_title(f'HI-tuned Model Attention\n({lang} Example)')
    ax2.set_xlabel('Key Tokens')
    ax2.set_ylabel('Query Tokens')
    
    plt.tight_layout()
    plt.savefig(f'results/attention_analysis/attention_heatmap_example_{example_idx}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_attention_summary(results):
    """Generate summary of attention analysis findings"""
    
    print("\nATTENTION ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Overall entropy comparison
    avg_en_model_entropy = np.mean(results['en_model_entropy'])
    avg_hi_model_entropy = np.mean(results['hi_model_entropy'])
    
    print(f"Average attention entropy:")
    print(f"  EN-tuned model: {avg_en_model_entropy:.3f}")
    print(f"  HI-tuned model: {avg_hi_model_entropy:.3f}")
    
    # Language-specific analysis
    languages = results['language']
    en_indices = [i for i, lang in enumerate(languages) if lang == 'English']
    hi_indices = [i for i, lang in enumerate(languages) if lang == 'Hindi']
    
    print(f"\nLanguage-specific patterns:")
    if en_indices:
        en_lang_en_model = np.mean([results['en_model_entropy'][i] for i in en_indices])
        en_lang_hi_model = np.mean([results['hi_model_entropy'][i] for i in en_indices])
        print(f"  English inputs: EN-model {en_lang_en_model:.3f}, HI-model {en_lang_hi_model:.3f}")
    
    if hi_indices:
        hi_lang_en_model = np.mean([results['en_model_entropy'][i] for i in hi_indices])
        hi_lang_hi_model = np.mean([results['hi_model_entropy'][i] for i in hi_indices])
        print(f"  Hindi inputs: EN-model {hi_lang_en_model:.3f}, HI-model {hi_lang_hi_model:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if avg_en_model_entropy > avg_hi_model_entropy:
        print("  EN-tuned model shows more diffuse attention (higher entropy)")
        print("  HI-tuned model shows more focused attention (lower entropy)")
    else:
        print("  HI-tuned model shows more diffuse attention (higher entropy)")
        print("  EN-tuned model shows more focused attention (lower entropy)")
    
    return {
        'avg_en_model_entropy': avg_en_model_entropy,
        'avg_hi_model_entropy': avg_hi_model_entropy,
        'language_specific': {
            'english': {'en_model': en_lang_en_model, 'hi_model': en_lang_hi_model} if en_indices else None,
            'hindi': {'en_model': hi_lang_en_model, 'hi_model': hi_lang_hi_model} if hi_indices else None
        }
    }

def main():
    """Run attention pattern analysis"""
    print("Starting Attention Pattern Analysis...")
    print("=" * 60)
    
    # Analyze patterns
    results, test_examples = analyze_attention_patterns()
    
    # Create visualizations
    plot_attention_entropy_comparison(results)
    
    # Plot attention heatmaps for first English and first Hindi example
    plot_attention_heatmap(results, test_examples, example_idx=0)  # English
    plot_attention_heatmap(results, test_examples, example_idx=2)  # Hindi
    
    # Generate summary
    summary = generate_attention_summary(results)
    
    # Save results
    import json
    with open('results/attention_analysis/attention_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAttention analysis complete!")
    print(f"Results saved to: results/attention_analysis/")

if __name__ == "__main__":
    main()
