#!/usr/bin/env python3

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Model setup
model_path = "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(
    model_path, trust_remote_code=True, dtype=torch.bfloat16
).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Example sequence (DNA)
sequence = "ATCGATCGATCG" * 40  # Make it ~480 chars like in the original

# Tokenize
encoding = tokenizer.encode_plus(
    sequence,
    return_tensors="pt",
    return_attention_mask=False,
    return_token_type_ids=False,
)
input_ids = encoding["input_ids"].to(device)

# Mask token at position 255 (like in original)
token_idx = 255
input_ids[0, token_idx] = tokenizer.mask_token_id

print(f"Input shape: {input_ids.shape}")
print(f"Masked token at position {token_idx}")

# Run inference
with torch.inference_mode():
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

print(f"Output logits shape: {logits.shape}")
print(f"Logits at masked position: {logits[0, token_idx, :5]}...")  # First 5 values
