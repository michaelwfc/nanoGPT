"""Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU 
https://huggingface.co/blog/trl-peft

Training a language model with RLHF typically involves the following three steps:
1. Fine-tune a pretrained LLM on a specific domain or corpus of instructions and human demonstrations
2. Collect a human annotated dataset and train a reward model
3. Further fine-tune the LLM from step 1 with the reward model and this dataset using RL (e.g. PPO)

Step 1: Load your active model in 8-bit precision
Step 2: Add extra trainable adapters using peft
Step 3: Use the same model to get the reference and active logits

"""