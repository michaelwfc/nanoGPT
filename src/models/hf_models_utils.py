import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import logging
logger = logging.getLogger(__name__)


def load_hf_model(model_name="EleutherAI/pythia-70m", model_path=None, local_files_only=False) -> AutoModelForCausalLM:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
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


# Load model onto the right device (GPU if available), and load tokenizer
def load_model(training_config, load_base_model=False):
    model_load_path = ""
    model_load_path = training_config["model"]["pretrained_name"]
    logger.debug(f"Loading default model: {model_load_path}")
    model = AutoModelForCausalLM.from_pretrained(model_load_path)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)

    logger.debug("Copying model to device")

    device_count = torch.cuda.device_count()
    if device_count > 0:
        logger.debug("Select GPU device")
        device = torch.device("cuda")
    else:
        logger.debug("Select CPU device")
        device = torch.device("cpu")

    model.to(device)

    logger.debug("Copying finished...")
    if "model_name" not in training_config:
        model_name = model_load_path
    else:
        model_name = training_config["model_name"]

    return model, tokenizer, device, model_name
