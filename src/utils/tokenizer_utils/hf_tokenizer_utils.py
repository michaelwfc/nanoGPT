"""run tokenizer"""
import random
import datasets

from pprint import pprint
from transformers import AutoTokenizer

import logging
logger = logging.getLogger(__name__)


def load_hf_tokenizer(model_name="EleutherAI/pythia-70m", tokenizer_path="./models/EleutherAI/pythia-70m"):
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


# Wrapper for data load, split, tokenize for training
def tokenize_and_split_data(training_config, tokenizer):
    initialized_config = initialize_config_and_logging(training_config)
    dataset_path = initialized_config["datasets"]["path"]
    use_hf = initialized_config["datasets"]["use_hf"]
    print("tokenize", use_hf, dataset_path)
    if use_hf:
        dataset = datasets.load_dataset(dataset_path)
    else:
        dataset = load_dataset(dataset_path, tokenizer)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset


# Tokenize and split data
def load_dataset(dataset_path, tokenizer):
    random.seed(42)
    finetuning_dataset_loaded = datasets.load_dataset("json", data_files=dataset_path, split="train")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = training_config["model"]["max_length"]
    tokenized_dataset = finetuning_dataset_loaded.map(
        get_tokenize_function(tokenizer, max_length),  # returns tokenize_function
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
    return split_dataset


# Get function for tokenization, based on config parameters
def get_tokenize_function(tokenizer, _max_length):

    def tokenize_function(examples):
        max_length = _max_length

        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token

        if "question" in examples and "answer" in examples:
            text = examples["question"][0] + examples["answer"][0]
        elif "input" in examples and "output" in examples:
            text = examples["input"][0] + examples["output"][0]
        else:
            text = examples["text"][0]

        # Run tokenizer on all the text (the input and the output)
        tokenized_inputs = tokenizer(
            text,

            # Return tensors in a numpy array (other options are pytorch or tf objects)
            return_tensors="np",

            # Padding type is to pad to the longest sequence in the batch (other option is to a certain max length, or no padding)
            padding=True,
        )

        # Calculate max length
        max_length = min(
            tokenized_inputs["input_ids"].shape[1],
            max_length
        )

        if tokenized_inputs["input_ids"].shape[1] > max_length:
            logger.warn(
                f"Truncating input from {tokenized_inputs['input_ids'].shape[1]} to {max_length}"
            )

        tokenizer.truncation_side = "left"

        tokenized_inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
        )

        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]

        return tokenized_inputs
    return tokenize_function


if __name__ == "__main__":
    run_tokenizer()
