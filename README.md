# Multilingual Tradeoffs in Finetuned Language Models

## Overview
This project investigates how fine tuning a language model on one language may affect its ability to handle other languages. This research follows XLM-RoBERTa, fine tuned on English and Hindi using the XNLI dataset. The XNLI Dataset is a benchmark dataset designed to evaluate language models across different languages. It explores the trade off that occurs when the language model fine tuned on English is run against Hindi, and vice versa. The goal is to understand if improving the performance in one language comes at the cost of another, what changes happen inside the model, and what kind of mistakes the model can make. The approach was to fine tune the same base model on hindi and english data separately, run experiments and analyze how and where it affects the model.

## Result 1: Asymmetric Language Bias
English finetuning creates stronger language specialization (+0.122 bias) compared to Hindi finetuning (+0.01 bias), revealing that different languages respond differently to identical training methods.

## Result 2: Late Layer Divergence
Models maintain 95%+ similarity in early layers (0-6) but diverge dramatically in final layers, with Hindi tuned model dropping to 42% similarity by layer 12, showing that language specialization happens in deeper layers while preserving shared multilingual knowledge.

## Result 3: Entailment Vulnerability
Cross language transfer catastrophically affects entailment detection (drops from 47% → 16% accuracy) while contradiction remains robust (69-88%), demonstrating that positive logical relationships are disproportionately sensitive to language interference.

