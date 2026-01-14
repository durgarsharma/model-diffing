#!/usr/bin/env python3
"""
Step 4: Activation Analysis - Internal Representation Changes
Measure where behavioral differences occur inside the models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
os.makedirs("results/activation_analysis", exist_ok=True)

def extract_layer_activations(model, inputs, tokenizer):
    """Extract hidden states from all layers"""
    model.eval()
    
    with torch.no_grad():
        # Get outputs with hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
        
        # Pool hidden states (mean across sequence length)
        pooled_states = []
        for layer_states in hidden_states:
            # Mean pooling across sequence dimension
            pooled = layer_states.mean(dim=1).cpu().numpy()  # (batch_size, hidden_size)
            pooled_states.append(pooled)
    
    return pooled_states

def create_probe_set(csv_file, num_samples=200):
    """Create probe set from test data"""
    df = pd.read_csv(csv_file).head(num_samples)
    
    probe_data = []
    for _, row in df.iterrows():
        probe_data.append({
            'premise': row['premise'],
            'hypothesis': row['hypothesis'],
            'label': row['label']
        })
    
    return probe_data

def compute_cka(X, Y):
    """Compute Centered Kernel Alignment between two representation matrices"""
    # Simple CKA using linear kernel (dot product)
    
    # Center the matrices
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute kernels
    K_X = np.dot(X, X.T)
    K_Y = np.dot(Y, Y.T)
    
    # Compute CKA
    numerator = np.trace(np.dot(K_X, K_Y))
    denominator = np.sqrt(np.trace(np.dot(K_X, K_X)) * np.trace(np.dot(K_Y, K_Y)))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def analyze_layer_differences(en_model_path, hi_model_path, probe_data, language):
    """Analyze layer-wise representation differences between models"""
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Load models
    en_model = AutoModelForSequenceClassification.from_pretrained(en_model_path).to(device)
    hi_model = AutoModelForSequenceClassification.from_pretrained(hi_model_path).to(device)
    
    print(f"Analyzing {language} representations...")
    print(f"Probe set size: {len(probe_data)}")
    
    # Extract activations for all samples
    en_layer_activations = [[] for _ in range(13)]  # XLM-R has 12 layers + embeddings
    hi_layer_activations = [[] for _ in range(13)]
    
    for i, sample in enumerate(probe_data):
        if i % 50 == 0:
            print(f"  Processing sample {i}/{len(probe_data)}")
        
        # Tokenize input
        inputs = tokenizer(
            sample['premise'], sample['hypothesis'],
            return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(device)
        
        # Extract activations
        en_states = extract_layer_activations(en_model, inputs, tokenizer)
        hi_states = extract_layer_activations(hi_model, inputs, tokenizer)
        
        # Store layer-wise activations
        for layer_idx in range(13):
            en_layer_activations[layer_idx].append(en_states[layer_idx][0])  # [0] for batch dim
            hi_layer_activations[layer_idx].append(hi_states[layer_idx][0])
    
    # Convert to numpy arrays
    en_layer_activations = [np.array(layer_acts) for layer_acts in en_layer_activations]
    hi_layer_activations = [np.array(layer_acts) for layer_acts in hi_layer_activations]
    
    # Compute CKA similarities
    cka_similarities = []
    for layer_idx in range(13):
        cka = compute_cka(en_layer_activations[layer_idx], hi_layer_activations[layer_idx])
        cka_similarities.append(cka)
        print(f"  Layer {layer_idx}: CKA = {cka:.3f}")
    
    return cka_similarities, en_layer_activations, hi_layer_activations

def plot_cka_analysis(en_cka, hi_cka):
    """Plot CKA similarity across layers for both languages"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    layers = list(range(13))
    layer_names = ['Embedding'] + [f'Layer {i}' for i in range(1, 13)]
    
    # Plot 1: CKA by layer for both languages
    ax1.plot(layers, en_cka, 'o-', label='English', color='#2E86AB', linewidth=2, markersize=8)
    ax1.plot(layers, hi_cka, 's-', label='Hindi', color='#A23B72', linewidth=2, markersize=8)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('CKA Similarity (EN-tuned vs HI-tuned)')
    ax1.set_title('Layer-wise Representation Similarity')
    ax1.set_xticks(layers[::2])
    ax1.set_xticklabels([layer_names[i] for i in layers[::2]], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Difference between languages
    lang_diff = np.array(en_cka) - np.array(hi_cka)
    colors = ['red' if x < 0 else 'blue' for x in lang_diff]
    ax2.bar(layers, lang_diff, color=colors, alpha=0.7)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('CKA Difference (English - Hindi)')
    ax2.set_title('Language-specific Representation Changes')
    ax2.set_xticks(layers[::2])
    ax2.set_xticklabels([layer_names[i] for i in layers[::2]], rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap
    heatmap_data = np.array([en_cka, hi_cka])
    im = ax3.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    ax3.set_xticks(layers[::2])
    ax3.set_xticklabels([layer_names[i] for i in layers[::2]], rotation=45)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['English', 'Hindi'])
    ax3.set_title('CKA Similarity Heatmap')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(layers)):
            if j % 2 == 0:  # Only annotate every other layer for readability
                text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                              ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im, ax=ax3)
    plt.tight_layout()
    plt.savefig('results/activation_analysis/cka_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_layer_change_magnitudes(en_cka, hi_cka):
    """Plot which layers changed most between models"""
    
    plt.figure(figsize=(12, 6))
    
    layers = list(range(13))
    layer_names = ['Embedding'] + [f'Layer {i}' for i in range(1, 13)]
    
    # Calculate "change magnitude" as 1 - CKA (lower CKA = more change)
    en_change = 1 - np.array(en_cka)
    hi_change = 1 - np.array(hi_cka)
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, en_change, width, label='English Input Change', 
                   color='#2E86AB', alpha=0.8)
    bars2 = plt.bar(x + width/2, hi_change, width, label='Hindi Input Change', 
                   color='#A23B72', alpha=0.8)
    
    plt.xlabel('Layer')
    plt.ylabel('Representation Change Magnitude (1 - CKA)')
    plt.title('Layer-wise Representation Changes Between EN-tuned and HI-tuned Models')
    plt.xticks(x[::2], [layer_names[i] for i in x[::2]], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight layers with biggest changes
    max_en_layer = np.argmax(en_change)
    max_hi_layer = np.argmax(hi_change)
    
    bars1[max_en_layer].set_color('#1a5c7a')
    bars2[max_hi_layer].set_color('#7a1b4b')
    
    plt.tight_layout()
    plt.savefig('results/activation_analysis/layer_changes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return max_en_layer, max_hi_layer

def main():
    """Run activation analysis"""
    print("Starting Activation Analysis...")
    print("=" * 60)
    
    # Create probe sets
    print("Creating probe sets...")
    en_probe = create_probe_set("data/dataset/xnli/en_test.csv", num_samples=100)  # Smaller for speed
    hi_probe = create_probe_set("data/dataset/xnli/hi_test.csv", num_samples=100)
    
    # Analyze English representations
    print("\nAnalyzing English representations...")
    en_cka, en_en_acts, en_hi_acts = analyze_layer_differences(
        "models/xnli_tuned/xnli_en_tuned", 
        "models/xnli_tuned/xnli_hi_tuned", 
        en_probe, 
        "English"
    )
    
    # Analyze Hindi representations  
    print("\nAnalyzing Hindi representations...")
    hi_cka, hi_en_acts, hi_hi_acts = analyze_layer_differences(
        "models/xnli_tuned/xnli_en_tuned",
        "models/xnli_tuned/xnli_hi_tuned",
        hi_probe,
        "Hindi" 
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_cka_analysis(en_cka, hi_cka)
    max_en_layer, max_hi_layer = plot_layer_change_magnitudes(en_cka, hi_cka)
    
    # Generate summary
    print(f"\nACTIVATION ANALYSIS SUMMARY:")
    print("=" * 40)
    print(f"Layers with biggest changes:")
    print(f"  English input: Layer {max_en_layer} (change: {1-en_cka[max_en_layer]:.3f})")
    print(f"  Hindi input: Layer {max_hi_layer} (change: {1-hi_cka[max_hi_layer]:.3f})")
    
    avg_en_cka = np.mean(en_cka)
    avg_hi_cka = np.mean(hi_cka)
    print(f"Average CKA similarity:")
    print(f"  English: {avg_en_cka:.3f}")
    print(f"  Hindi: {avg_hi_cka:.3f}")
    
    # Save results
    results = {
        'english_cka': en_cka,
        'hindi_cka': hi_cka,
        'max_change_layers': {'english': int(max_en_layer), 'hindi': int(max_hi_layer)},
        'average_similarity': {'english': float(avg_en_cka), 'hindi': float(avg_hi_cka)}
    }
    
    import json
    with open('results/activation_analysis/cka_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/activation_analysis/")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
