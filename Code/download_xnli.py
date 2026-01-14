#!/usr/bin/env python3
"""
Step 1.1: Download and explore XNLI dataset (FINAL VERSION)
"""

import pandas as pd
from datasets import load_dataset
import os

# Create directory structure
os.makedirs("data/xnli", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("📥 Downloading XNLI dataset...")

# Load XNLI dataset with all languages config
xnli_dataset = load_dataset("xnli", "all_languages")

print("✅ Dataset downloaded successfully!")
print("\n📊 Dataset structure:")
print(f"Available splits: {list(xnli_dataset.keys())}")

# Examine the dataset
for split_name, split_data in xnli_dataset.items():
    print(f"\n{split_name.upper()} split:")
    print(f"  - Size: {len(split_data)} examples")

print("\n🔍 Understanding data structure:")
test_data = xnli_dataset['test']

# Helper function to get hypothesis in specific language
def get_hypothesis_by_language(hypothesis_dict, target_language):
    """Extract hypothesis for a specific language"""
    languages = hypothesis_dict['language']
    translations = hypothesis_dict['translation']
    
    # Find index of target language
    try:
        lang_index = languages.index(target_language)
        return translations[lang_index]
    except ValueError:
        return None

# Check sample data
sample = test_data[0]
print("Sample data:")
print(f"English premise: {sample['premise']['en']}")
print(f"Hindi premise: {sample['premise']['hi']}")
print(f"English hypothesis: {get_hypothesis_by_language(sample['hypothesis'], 'en')}")
print(f"Hindi hypothesis: {get_hypothesis_by_language(sample['hypothesis'], 'hi')}")
print(f"Label: {sample['label']}")

print("\n📝 More sample English and Hindi data:")

# Extract samples for both languages
def extract_language_samples(dataset_split, language, num_samples=3):
    """Extract data for a specific language"""
    samples = []
    for i in range(min(num_samples, len(dataset_split))):
        example = dataset_split[i]
        samples.append({
            'premise': example['premise'][language],
            'hypothesis': get_hypothesis_by_language(example['hypothesis'], language),
            'label': example['label']
        })
    return samples

en_samples = extract_language_samples(test_data, 'en', 3)
hi_samples = extract_language_samples(test_data, 'hi', 3)

print("\nEnglish samples:")
for i, sample in enumerate(en_samples):
    print(f"Sample {i+1}:")
    print(f"  Premise: {sample['premise']}")
    print(f"  Hypothesis: {sample['hypothesis']}")
    print(f"  Label: {sample['label']}")
    print()

print("Hindi samples:")
for i, sample in enumerate(hi_samples):
    print(f"Sample {i+1}:")
    print(f"  Premise: {sample['premise']}")
    print(f"  Hypothesis: {sample['hypothesis']}")
    print(f"  Label: {sample['label']}")
    print()

# Create language-specific datasets
print("💾 Creating language-specific datasets...")

def create_language_dataset(dataset_split, language):
    """Create a dataset for a specific language"""
    data = []
    for example in dataset_split:
        data.append({
            'premise': example['premise'][language],
            'hypothesis': get_hypothesis_by_language(example['hypothesis'], language),
            'label': example['label']
        })
    return pd.DataFrame(data)

# Create test sets
en_test = create_language_dataset(test_data, 'en')
hi_test = create_language_dataset(test_data, 'hi')

print(f"English test samples: {len(en_test)}")
print(f"Hindi test samples: {len(hi_test)}")

# Save test sets
en_test.to_csv("data/xnli/en_test.csv", index=False)
hi_test.to_csv("data/xnli/hi_test.csv", index=False)

# Create training sets (we'll need these for fine-tuning)
print("💾 Creating training sets...")
train_data = xnli_dataset['train']

en_train = create_language_dataset(train_data, 'en')
hi_train = create_language_dataset(train_data, 'hi')

# Save smaller training sets (first 5000 examples for quick training)
en_train_small = en_train.head(5000)
hi_train_small = hi_train.head(5000)

en_train_small.to_csv("data/xnli/en_train.csv", index=False)
hi_train_small.to_csv("data/xnli/hi_train.csv", index=False)

print("\n✅ Step 1.1 Complete!")
print("Files saved:")
print("  - data/xnli/en_test.csv")
print("  - data/xnli/hi_test.csv")
print("  - data/xnli/en_train.csv (5000 samples)")
print("  - data/xnli/hi_train.csv (5000 samples)")

# Basic statistics
print(f"\n📈 Quick stats:")
print(f"English test set: {len(en_test)} examples")
print(f"Hindi test set: {len(hi_test)} examples")
print(f"English train set: {len(en_train_small)} examples")
print(f"Hindi train set: {len(hi_train_small)} examples")

# Label distribution
labels = ['entailment', 'neutral', 'contradiction']
print(f"\nLabel distribution (English test): {en_test['label'].value_counts().to_dict()}")
print(f"Label distribution (Hindi test): {hi_test['label'].value_counts().to_dict()}")

# Verify data quality
print(f"\n🔍 Data quality check:")
print(f"English test - any missing hypotheses: {en_test['hypothesis'].isna().sum()}")
print(f"Hindi test - any missing hypotheses: {hi_test['hypothesis'].isna().sum()}")