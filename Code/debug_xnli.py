#!/usr/bin/env python3
"""
Quick debug script to understand XNLI structure
"""

from datasets import load_dataset

# Load XNLI dataset
xnli_dataset = load_dataset("xnli", "all_languages")
test_data = xnli_dataset['test']

# Check first example structure
sample = test_data[0]

print("=== PREMISE STRUCTURE ===")
print(f"Type: {type(sample['premise'])}")
print(f"Keys: {sample['premise'].keys()}")
print(f"English premise: {sample['premise']['en']}")
print(f"Hindi premise: {sample['premise']['hi']}")

print("\n=== HYPOTHESIS STRUCTURE ===")  
print(f"Type: {type(sample['hypothesis'])}")
print(f"Keys: {sample['hypothesis'].keys()}")
print(f"Content: {sample['hypothesis']}")

# Let's check a few more examples to understand the pattern
print("\n=== CHECKING MULTIPLE EXAMPLES ===")
for i in range(3):
    sample = test_data[i]
    print(f"\nExample {i+1}:")
    print(f"Hypothesis structure: {sample['hypothesis']}")
