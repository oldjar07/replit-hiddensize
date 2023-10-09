---
license: cc-by-sa-4.0
---

# Replit Code V-1.5 3B

Developed by: Replit, Inc.

## Model Description

Replit Code v1.5 is a 3.3B parameter Causal Language Model focused on **Code Completion**.

The model is trained on 1T tokens of code (~200B tokens  over 5 epochs, including linear cooldown) for 30 programming languages from a subset of permissively licensed code from Bigcode's [Stack Dedup V2 dataset](https://huggingface.co/datasets/bigcode/the-stack-dedup).

We use the GPTNeoX tokenizer with a custom trained and optimized vocabulary of 32768 tokens that led to single-digit % points on compression while maintaining or improving coverage on our training corpus.

The model has been trained on the [MosaicML](https://www.mosaicml.com/) platform on 128  H100-80GB GPUs.

The model has a context window size of 4096 tokens.

## Dependancies
You will need to install the latest versions of the following dependencies:
```
einops
torch
transformers
```

## How to Use (Coming Soon!)

(Details and code examples coming soon!)

## Intended Use

Replit intends this model be used by anyone as a foundational model for application-specific fine-tuning without strict limitations on commercial use.

The model is trained specifically for code completion tasks.

## Limitations
The pre-training dataset may have contained offensive or inappropriate content even after applying data cleansing filters, and such content may be reflected in model generated text. We recommend that users exercise reasonable caution when using in production systems. Do not use for any applications that may cause harm or distress to individuals or groups.
