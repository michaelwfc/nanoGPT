"""run tokenizer"""
import pandas as pd
import datasets

from pprint import pprint
from transformers import AutoTokenizer


def load_hf_tokenizer(model_name="EleutherAI/pythia-70m", tokenizer_path="./models/EleutherAI/pythia-70m" ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_tokenizer():
    text = "Hi, how are you?"
    list_texts = ["Hi, how are you?", "I'm good", "Yes"]

    tokenizer = load_hf_tokenizer()

    print(f"vocab_size={tokenizer.vocab_size}\nsplit_special_tokens={tokenizer.split_special_tokens}\n\
          eos_token={tokenizer.eos_token}\npad_token={tokenizer.pad_token}\n")
    tokenizer.chat_template

    encoded_text = tokenizer(text)["input_ids"]

    decoded_text = tokenizer.decode(encoded_text)
    print("Decoded tokens back into text: ", decoded_text)

    list_texts = ["Hi, how are you?", "I'm good", "Yes"]
    encoded_texts = tokenizer(list_texts)
    print("Encoded several texts: ", encoded_texts["input_ids"])

    tokenizer.pad_token = tokenizer.eos_token
    encoded_texts_longest = tokenizer(list_texts, padding=True)
    print("Using padding: ", encoded_texts_longest["input_ids"])

    encoded_texts_truncation = tokenizer(list_texts, max_length=3, truncation=True)
    print("Using truncation: ", encoded_texts_truncation["input_ids"])

    tokenizer.truncation_side = "left"
    encoded_texts_truncation_left = tokenizer(list_texts, max_length=3, truncation=True)
    print("Using left-side truncation: ", encoded_texts_truncation_left["input_ids"])

    encoded_texts_both = tokenizer(list_texts, max_length=3, truncation=True, padding=True)
    print("Using both padding and truncation: ", encoded_texts_both["input_ids"])


if __name__ == "__main__":
    run_tokenizer()
