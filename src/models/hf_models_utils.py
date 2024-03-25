import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def load_hf_model(model_name="EleutherAI/pythia-70m", model_path=None):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model


def load_hf_pipeline(model="meta-llama/Llama-2-7b-chat-hf", cache_dir="./models/meta-llama/Llama-2-7b-chat-hf"):
    """huggingface-cli login """

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
