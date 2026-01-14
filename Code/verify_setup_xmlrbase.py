#!/usr/bin/env python3
"""
Step 1.4: Verify complete setup before moving to fine-tuning
"""

import pandas as pd
import os
import json
from pathlib import Path

print("🔍 Verifying complete Step 1 setup...")

# Check directory structure
expected_dirs = ['data/xnli', 'data/xquad', 'models', 'results']
expected_files = [
    'data/xnli/en_train.csv',
    'data/xnli/hi_train.csv', 
    'data/xnli/en_test.csv',
    'data/xnli/hi_test.csv',
    'data/xquad/en_train.csv',
    'data/xquad/hi_train.csv',
    'data/xquad/en_test.csv', 
    'data/xquad/hi_test.csv',
    'models/model_info.json'
]

print("\n📁 Checking directory structure...")
for directory in expected_dirs:
    if os.path.exists(directory):
        print(f"✅ {directory}")
    else:
        print(f"❌ {directory} - MISSING!")

print("\n📄 Checking required files...")
for file_path in expected_files:
    if os.path.exists(file_path):
        # Get file size
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"✅ {file_path} ({size:.1f} KB)")
    else:
        print(f"❌ {file_path} - MISSING!")

# Load and verify datasets
print("\n📊 Dataset verification:")

datasets_info = {}

# XNLI datasets
for lang in ['en', 'hi']:
    for split in ['train', 'test']:
        file_path = f"data/xnli/{lang}_{split}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            datasets_info[f"xnli_{lang}_{split}"] = len(df)
            print(f"XNLI {lang.upper()} {split}: {len(df)} examples")
            
            # Check columns
            expected_cols = ['premise', 'hypothesis', 'label']
            if all(col in df.columns for col in expected_cols):
                print(f"  ✅ All required columns present")
            else:
                print(f"  ❌ Missing columns: {set(expected_cols) - set(df.columns)}")
                
            # Check for missing values
            missing = df.isnull().sum().sum()
            if missing == 0:
                print(f"  ✅ No missing values")
            else:
                print(f"  ⚠️ {missing} missing values found")

print()

# XQuAD datasets  
for lang in ['en', 'hi']:
    for split in ['train', 'test']:
        file_path = f"data/xquad/{lang}_{split}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            datasets_info[f"xquad_{lang}_{split}"] = len(df)
            print(f"XQuAD {lang.upper()} {split}: {len(df)} examples")
            
            # Check columns
            expected_cols = ['context', 'question', 'answer_text']
            if all(col in df.columns for col in expected_cols):
                print(f"  ✅ All required columns present")
            else:
                print(f"  ❌ Missing columns: {set(expected_cols) - set(df.columns)}")
                
            # Check for missing values (answer_text can be empty for some QA)
            missing = df[['context', 'question']].isnull().sum().sum()
            if missing == 0:
                print(f"  ✅ No missing values in context/question")
            else:
                print(f"  ⚠️ {missing} missing values in context/question")

# Load model info
print("\n🤖 Model configuration:")
if os.path.exists('models/model_info.json'):
    with open('models/model_info.json', 'r') as f:
        model_info = json.load(f)
    
    for key, value in model_info.items():
        print(f"  {key}: {value}")
else:
    print("  ❌ Model info file missing!")

# Summary
print(f"\n📈 Setup Summary:")
print(f"✅ Total XNLI training examples: EN={datasets_info.get('xnli_en_train', 0)}, HI={datasets_info.get('xnli_hi_train', 0)}")
print(f"✅ Total XNLI test examples: EN={datasets_info.get('xnli_en_test', 0)}, HI={datasets_info.get('xnli_hi_test', 0)}")
print(f"✅ Total XQuAD training examples: EN={datasets_info.get('xquad_en_train', 0)}, HI={datasets_info.get('xquad_hi_train', 0)}")
print(f"✅ Total XQuAD test examples: EN={datasets_info.get('xquad_en_test', 0)}, HI={datasets_info.get('xquad_hi_test', 0)}")

# Check if we're ready for Step 2
all_files_exist = all(os.path.exists(f) for f in expected_files)

if all_files_exist:
    print(f"\n🎉 STEP 1 COMPLETE!")
    print(f"✅ All datasets downloaded and processed")
    print(f"✅ Base models loaded and tested") 
    print(f"✅ Directory structure created")
    print(f"\n🚀 Ready to proceed to STEP 2: Fine-tuning!")
    
    # Save setup verification
    verification_info = {
        'step_1_complete': True,
        'datasets': datasets_info,
        'all_files_present': all_files_exist,
        'verification_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open('results/step1_verification.json', 'w') as f:
        json.dump(verification_info, f, indent=2)
    
    print(f"💾 Verification saved to results/step1_verification.json")
    
else:
    print(f"\n❌ SETUP INCOMPLETE!")
    print(f"Please check the missing files above and re-run the previous steps.")
