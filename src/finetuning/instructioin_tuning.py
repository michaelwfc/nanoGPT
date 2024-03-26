"""intruction tuning from deeplearning.ai

https://learn.deeplearning.ai/courses/finetuning-large-language-models/lesson/4/instruction-finetuning"""
import os
import pandas as pd

from pprint import pprint
import torch
# from llama import BasicModelRunner
from lamini import BasicModelRunner, LlamaV2Runner
from datasets import Dataset, DatasetDict, load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from configs.train_configs import get_training_args
from utils.torch_utils import get_device, get_model_flops
from utils.data_utils import load_instruction_tuned_dataset, load_jsonlines_file
from utils.tokenizer_utils.hf_tokenizer_utils import load_hf_tokenizer
from models.hf_models_utils import load_hf_model
from trainner import Trainer


class InstructonTunner():
    def __init__(self, model_name="EleutherAI/pythia-70m", model_path="./download_models/EleutherAI/pythia-70m") -> None:

        self.model_name = model_name
        self.model_path = model_path
        self.dataset_name = "tatsu-lab/alpaca"
        self.data_save_dir = "./data/tatsu-lab/alpaca"
        self.processed_data_path = './data/tatsu-lab/alpaca/alpaca_processed.jsonl'
        self.max_length = 128
        self.train_steps = 3

        self.train_args = get_training_args()
        self.save_dir = f"output/alpaca"

        self.device = get_device()

        self.model_flops = get_model_flops(max_length=self.max_length, gradient_accumulation_steps=self.train_args.gradient_accumulation_steps)
        self.tokenizer = load_hf_tokenizer(tokenizer_path=model_path)
        self.base_model = load_hf_model(model_name=model_name, model_path=model_path)
        self.input_key = None
        self.ouput_key = None
        self.suffix = "ids"

    def _load_instruction_tuned_dataset(self) -> Dataset:
        # Load instruction tuned dataset
        if os.path.exists(self.processed_data_path):
            data_ls = load_jsonlines_file(self.processed_data_path)
        else:
            data_ls = load_instruction_tuned_dataset(dataset_name=self.dataset_name, save_path=self.processed_data_path)

        dataset_dict = load_dataset('json', data_files=self.processed_data_path)  # ,split='train')
        dataset = dataset_dict["train"]

        self.input_key = 'input'
        self.output_key = 'output'

        self.input_ids_key = f"{self.input_key}_{self.suffix}"
        self.output_ids_key = f"{self.output_key}_{self.suffix}"
        return dataset

    def prepare_data(self):
        """fineturn data: High quality, Diversity, Real, More

        1. collect instruction-response pairs
        2. concatenate paris(add prompt template if applicable)
        3. tokenize:pad, turncate
        4. split into train/test

        Args:
            data: _description_
        """

        dataset = self._load_instruction_tuned_dataset()

        tokenized_dataset = dataset.map(self._tokenizing, batched=True, batch_size=4, drop_last_batch=True)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
        return split_dataset

    def _tokenizing(self, example):
        input_text = example[self.input_key]
        output_text = example[self.output_key]
        tokenized_inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )

        tokenized_outputs = self.tokenizer(
            output_text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )

        example[self.input_ids_key] = tokenized_inputs
        example[self.output_ids_key] = tokenized_outputs

        return tokenized_inputs

    def train(self):
        train_dataset, eval_dataset = self.prepare_data()

        trainer = Trainer(model=self.base_model, model_flops=self.model_flops,
                          total_steps=self.train_steps, args=self.train_args,
                          train_dataset=train_dataset, eval_dataset=eval_dataset,
                          tokenizer=self.tokenizer)
        training_output = trainer.train()

        trainer.save_model(self.save_dir)
        print("Saved model to:", self.save_dir)


    @staticmethod
    def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
        # Tokenize
        input_ids = tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        )

        # Generate
        device = model.device
        generated_tokens_with_prompt = model.generate(
            input_ids=input_ids.to(device),
            max_length=max_output_tokens
        )

        # Decode
        generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

        # Strip the prompt
        generated_text_answer = generated_text_with_prompt[0][len(text):]

        return generated_text_answer


def run_non_instruct_model():
    non_instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-hf", local_cache_file="./models/meta-llama/Llama-2-7b-hf")
    non_instruct_output = non_instruct_model("Tell me how to train my dog to sit")
    print("Not instruction-tuned output (Llama 2 Base):", non_instruct_output)


if __name__ == "__main__":
    instruction_tunner = InstructonTunner()
    instruction_tunner.prepare_data()
    # run_non_instruct_model()
    load_hf_model()
