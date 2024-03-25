"""the utils for data"""
from pprint import pprint
import itertools
import jsonlines
from typing import List, Dict
from datasets import load_dataset, Dataset
import pandas as pd


alpaca_dataset = "tatsu-lab/alpaca"
lamini_docs_dataset = "lamini/lamini_docs"
taylor_swift_dataset = "lamini/taylor_swift"
bts_dataset = "lamini/bts"
open_llms = "lamini/open_llms"


prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def load_instruction_tuned_dataset(dataset_name="tatsu-lab/alpaca", save_path='./data/tatsu-lab/alpaca/alpaca_processed.jsonl') -> List[Dict]:
    # Load instruction tuned dataset
    instruction_tuned_dataset = load_dataset(dataset_name, split="train", streaming=True)
    m = 100
    print("Instruction-tuned dataset:")
    top_m = list(itertools.islice(instruction_tuned_dataset, m))
    # for j in top_m:
    #     print(j)

    processed_data = []
    for j in top_m:
        if not j["input"]:
            processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])
        else:
            processed_prompt = prompt_template_with_input.format(instruction=j["instruction"], input=j["input"])

        processed_data.append({"input": processed_prompt, "output": j["output"]})

    pprint(processed_data[0])

    # Save data to jsonl
    with jsonlines.open(save_path, 'w') as writer:
        writer.write_all(processed_data)
    return processed_data


def load_jsonlines_file(path='./data/tatsu-lab/alpaca_processed.jsonl') -> List[Dict]:
    with jsonlines.open(path, 'r') as reader:
        data = [obj for obj in reader]
    return data


def load_lamini_data(filename="lamini_docs.jsonl"):

    instruction_dataset_df = pd.read_json(filename, lines=True)
    examples = instruction_dataset_df.to_dict()

    if "question" in examples and "answer" in examples:
        text = examples["question"][0] + examples["answer"][0]
    elif "instruction" in examples and "response" in examples:
        text = examples["instruction"][0] + examples["response"][0]
    elif "input" in examples and "output" in examples:
        text = examples["input"][0] + examples["output"][0]
    else:
        text = examples["text"][0]

    prompt_template = """### Question:
        {question}

        ### Answer:"""

    num_examples = len(examples["question"])
    finetuning_dataset = []
    for i in range(num_examples):
        question = examples["question"][i]
        answer = examples["answer"][i]
        text_with_prompt_template = prompt_template.format(question=question)
        finetuning_dataset.append({"question": text_with_prompt_template, "answer": answer})

    print("One datapoint in the finetuning dataset:")
    pprint(finetuning_dataset[0])
    return finetuning_dataset


if __name__ == "__main__":
    dataset = load_jsonlines_file()
