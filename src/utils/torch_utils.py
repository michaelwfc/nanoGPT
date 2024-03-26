import torch
import logging
from transformers import AutoModelForCausalLM


logger = logging.getLogger(__name__)


def get_device():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        logger.debug("Select GPU device")
        device = torch.device("cuda")
    else:
        logger.debug("Select CPU device")
        device = torch.device("cpu")
    return device


def get_model_flops(base_model: AutoModelForCausalLM, max_length, gradient_accumulation_steps):
    model_flops = (
        base_model.floating_point_ops(
            {
                "input_ids": torch.zeros(
                    (1, max_length)
                )
            }
        )
        * gradient_accumulation_steps
    )

    # print(base_model)
    print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
    print("Flops", model_flops / 1e9, "GFLOPs")
