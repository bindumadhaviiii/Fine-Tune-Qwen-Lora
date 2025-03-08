# utils.py - Helper functions for dataset handling.

from transformers import AutoTokenizer

def preprocess_text(text, model_name="Qwen/Qwen1.5-0.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)
