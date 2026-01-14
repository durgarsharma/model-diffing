#!/usr/bin/env python3
"""
Step 1.3: Setup Base Model (XLM-R-base) and test it
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import pandas as pd
import os

# Create models directory
os.makedirs("models", exist_ok=True)

print("🤖 Setting up XLM-RoBERTa-base model...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Use CPU for this project to keep it simple and fast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer (this will download the model files)
print("\n📥 Loading tokenizer...")
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("✅ Tokenizer loaded successfully!")
print(f"Vocab size: {len(tokenizer)}")
print(f"Max length: {tokenizer.model_max_length}")

# Test tokenization with English and Hindi text
print("\n🔤 Testing tokenization:")

# Test sentences
en_text = "Hello, how are you today?"
hi_text = "नमस्ते, आप कैसे हैं?"

en_tokens = tokenizer.tokenize(en_text)
hi_tokens = tokenizer.tokenize(hi_text)

print(f"English: '{en_text}'")
print(f"  Tokens: {en_tokens}")
print(f"  Token IDs: {tokenizer.encode(en_text)}")

print(f"\nHindi: '{hi_text}'")
print(f"  Tokens: {hi_tokens}")
print(f"  Token IDs: {tokenizer.encode(hi_text)}")

# Load models for both tasks
print("\n🏗️ Loading base models for both tasks...")

# For XNLI (classification)
print("Loading model for XNLI (sequence classification)...")
model_nli = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3  # entailment, neutral, contradiction
)
model_nli.to(device)

# For XQuAD (question answering)
print("Loading model for XQuAD (question answering)...")
model_qa = AutoModelForQuestionAnswering.from_pretrained(model_name)
model_qa.to(device)

print("✅ Both models loaded successfully!")

# Test the models with sample data
print("\n🧪 Testing models with sample data...")

# Load sample data
en_nli = pd.read_csv("data/xnli/en_test.csv").head(1)
hi_nli = pd.read_csv("data/xnli/hi_test.csv").head(1)
en_qa = pd.read_csv("data/xquad/en_test.csv").head(1)
hi_qa = pd.read_csv("data/xquad/hi_test.csv").head(1)

# Test NLI model
print("\n📊 Testing NLI model:")

def test_nli_model(model, tokenizer, premise, hypothesis, language):
    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=-1).item()
    
    labels = ['entailment', 'neutral', 'contradiction']
    print(f"{language} NLI:")
    print(f"  Premise: {premise[:100]}...")
    print(f"  Hypothesis: {hypothesis[:100]}...")
    print(f"  Predicted: {labels[predicted_label]} (confidence: {predictions[0][predicted_label]:.3f})")
    
    return predicted_label

# Test with English
en_pred = test_nli_model(
    model_nli, tokenizer,
    en_nli.iloc[0]['premise'],
    en_nli.iloc[0]['hypothesis'],
    "English"
)

# Test with Hindi  
hi_pred = test_nli_model(
    model_nli, tokenizer,
    hi_nli.iloc[0]['premise'], 
    hi_nli.iloc[0]['hypothesis'],
    "Hindi"
)

# Test QA model
print("\n❓ Testing QA model:")

def test_qa_model(model, tokenizer, context, question, language):
    inputs = tokenizer(
        question,
        context, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()
        
        if end_idx >= start_idx:
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        else:
            answer = "[No answer found]"
    
    print(f"{language} QA:")
    print(f"  Question: {question}")
    print(f"  Context: {context[:150]}...")
    print(f"  Predicted Answer: {answer}")
    
    return answer

# Test with English
en_qa_pred = test_qa_model(
    model_qa, tokenizer,
    en_qa.iloc[0]['context'],
    en_qa.iloc[0]['question'], 
    "English"
)

# Test with Hindi
hi_qa_pred = test_qa_model(
    model_qa, tokenizer,
    hi_qa.iloc[0]['context'],
    hi_qa.iloc[0]['question'],
    "Hindi"
)

print("\n✅ Step 1.3 Complete!")
print("\nModel setup summary:")
print(f"✅ XLM-RoBERTa-base tokenizer loaded")
print(f"✅ NLI model loaded (3 labels)")
print(f"✅ QA model loaded") 
print(f"✅ Both models tested on EN/HI data")
print(f"✅ Device: {device}")

print(f"\nNext: We're ready to start fine-tuning!")
print(f"📁 We have datasets for both tasks in EN/HI")
print(f"🤖 We have working base models")
print(f"⚡ Ready to create EN-tuned and HI-tuned variants")

# Save model info for later steps
model_info = {
    'model_name': model_name,
    'device': str(device),
    'tokenizer_vocab_size': len(tokenizer),
    'max_length': tokenizer.model_max_length
}

import json
with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n💾 Model info saved to models/model_info.json")
