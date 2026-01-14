#!/usr/bin/env python3
"""
Step 1.2: Download and explore XQuAD dataset
"""

import pandas as pd
from datasets import load_dataset
import os

# Create directory for XQuAD
os.makedirs("data/xquad", exist_ok=True)

print("📥 Downloading XQuAD dataset...")

# Load XQuAD dataset - need to specify specific language configs
print("Loading English XQuAD...")
xquad_en = load_dataset("xquad", "xquad.en")

print("Loading Hindi XQuAD...")  
xquad_hi = load_dataset("xquad", "xquad.hi")

print("✅ XQuAD datasets downloaded successfully!")

print(f"\n📊 XQuAD English structure:")
print(f"Available splits: {list(xquad_en.keys())}")
for split_name, split_data in xquad_en.items():
    print(f"{split_name.upper()}: {len(split_data)} examples")

print(f"\n📊 XQuAD Hindi structure:")
print(f"Available splits: {list(xquad_hi.keys())}")
for split_name, split_data in xquad_hi.items():
    print(f"{split_name.upper()}: {len(split_data)} examples")

# Examine sample data
print("\n🔍 Sample data structure:")

en_sample = xquad_en['validation'][0]
hi_sample = xquad_hi['validation'][0]

print("English sample fields:")
for key, value in en_sample.items():
    print(f"  {key}: {type(value)}")

print(f"\nEnglish example:")
print(f"Context: {en_sample['context'][:200]}...")
print(f"Question: {en_sample['question']}")
print(f"Answer: {en_sample['answers']}")

print(f"\nHindi example:")
print(f"Context: {hi_sample['context'][:200]}...")
print(f"Question: {hi_sample['question']}")
print(f"Answer: {hi_sample['answers']}")

# Process and save the datasets
print("\n💾 Processing and saving XQuAD datasets...")

def process_xquad_data(dataset_split):
    """Process XQuAD data into a clean format"""
    data = []
    for example in dataset_split:
        # Extract the first answer (XQuAD typically has one answer per question)
        answer_text = example['answers']['text'][0] if example['answers']['text'] else ""
        answer_start = example['answers']['answer_start'][0] if example['answers']['answer_start'] else 0
        
        data.append({
            'id': example['id'],
            'context': example['context'],
            'question': example['question'],
            'answer_text': answer_text,
            'answer_start': answer_start
        })
    
    return pd.DataFrame(data)

# Process validation sets (XQuAD only has validation split)
en_xquad = process_xquad_data(xquad_en['validation'])
hi_xquad = process_xquad_data(xquad_hi['validation'])

print(f"English XQuAD: {len(en_xquad)} examples")
print(f"Hindi XQuAD: {len(hi_xquad)} examples")

# Save processed datasets
en_xquad.to_csv("data/xquad/en_xquad.csv", index=False)
hi_xquad.to_csv("data/xquad/hi_xquad.csv", index=False)

# Since XQuAD doesn't have training data, we'll create small training sets from the validation data
# Split each language dataset: first 800 for training, rest for testing
print("\n📊 Creating train/test splits from validation data...")

en_train_xquad = en_xquad.head(800)
en_test_xquad = en_xquad.tail(len(en_xquad) - 800)

hi_train_xquad = hi_xquad.head(800) 
hi_test_xquad = hi_xquad.tail(len(hi_xquad) - 800)

# Save splits
en_train_xquad.to_csv("data/xquad/en_train.csv", index=False)
en_test_xquad.to_csv("data/xquad/en_test.csv", index=False)
hi_train_xquad.to_csv("data/xquad/hi_train.csv", index=False)
hi_test_xquad.to_csv("data/xquad/hi_test.csv", index=False)

print("\n✅ Step 1.2 Complete!")
print("XQuAD files saved:")
print("  - data/xquad/en_train.csv")
print("  - data/xquad/en_test.csv")
print("  - data/xquad/hi_train.csv")
print("  - data/xquad/hi_test.csv")
print("  - data/xquad/en_xquad.csv (full dataset)")
print("  - data/xquad/hi_xquad.csv (full dataset)")

print(f"\n📈 Final stats:")
print(f"English train: {len(en_train_xquad)} examples")
print(f"English test: {len(en_test_xquad)} examples")
print(f"Hindi train: {len(hi_train_xquad)} examples") 
print(f"Hindi test: {len(hi_test_xquad)} examples")

# Sample questions for verification
print(f"\n📝 Sample questions:")
print(f"English: {en_test_xquad.iloc[0]['question']}")
print(f"Hindi: {hi_test_xquad.iloc[0]['question']}")
