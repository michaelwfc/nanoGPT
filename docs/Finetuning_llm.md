
# RLHF : Reinforcement Learning from human feedback

## step 1. Supervied fine-tuning(SFT): 

Fine-tune a pretrained LLM(GPT-3) on a specific domain or corpus of instructions and human demonstrations  using supervised learning on Demonstration data 

Base LLM:  Llama2, BLOOMZ, Flan-T5, Flan-UL2, and OPT-IML

## step 2. Train a Reward Model on comparison data

 Collect a human annotated dataset and train a reward model

## step 3. Further fine-tune the LLM from step 1 with the reward model and this dataset using RL (e.g. PPO)

fine-tune our supervised learning baseline to maximize this reward using the PPO algorithm
PPO model initialized from the supervised policy, a new prompt is sampled from the dataset, the policy generate output,  which get a reward from the RM. the reward is uesd to update the policy using PPO


# Reference

## Courses

[Andrej Karpathy-Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
[finetuning-large-language-models](https://learn.deeplearning.ai/courses/finetuning-large-language-models/)
[Fine-tuning LLMs with PEFT and LoRA](https://www.youtube.com/watch?v=Us5ZFp16PaU&t=232s)
[OpenBMB] 大模型公开课
[Hyung Won Chung- Instruction finetuning and RLHF lecture](https://www.youtube.com/watch?v=zjrM-MW-0y0)
[Stanford CS224N-Lecture 11: Prompting, Instruction Finetuning, and RLHF]

## Tutorial

- [gpt-dev.ipynb](https://colab.research.google.com/drive/13QHBS__yCPQI8B_wVwVV0b05_gCH-7Pq)

- [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)

- [LLAMA-2 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA](https://www.youtube.com/watch?v=LslC2nKEEGU&list=PLVEEucA9MYhMkc4HvgHw-TvycgoMhADOI)
- [LLAMA-3](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3)

- [huggingface-peft](https://huggingface.co/blog/peft)

- [DeepLearningAI-Efficient Fine-Tuning for Llama-v2-7b on a Single GPU](https://www.youtube.com/watch?v=g68qlo9Izf0)

- [finetune 的几种方法和对应框架 ](https://note.iawen.com/note/llm/finetune)
- [LLMs Fine-tuning 学习笔记（一）：trl+peft](https://www.cnblogs.com/lokvahkoor/p/17413273.html)


- BarraHome/Mistroll-7B-v2.2  
This model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.
- [Fine-tuning Llama 2 on a Custom Dataset]（https://www.mlexpert.io/blog/fine-tuning-llama-2-on-custom-dataset）
- [awesome-llms-fine-tuning](https://github.com/Curated-Awesome-Lists/awesome-llms-fine-tuning)
- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#4-fine-tune-llm-using-trl-and-the-sfttrainer)
 

## Open source prjects:

[fastchat-training](https://github.com/lm-sys/FastChat/blob/main/docs/training.md)
[mistral-finetune](https://github.com/mistralai/mistral-finetune)
[skypilot](https://github.com/skypilot-org/skypilot)
[alpaca-lora](https://github.com/tloen/alpaca-lora)
[lamini](https://github.com/lamini-ai/lamini)
[litgpt](https://github.com/Lightning-AI/litgpt)

[ai2-RL4LMs](https://github.com/allenai/RL4LMs)

# tokenization

[google-sentecepice]()
[openai-tiktoken]



## LORA

LoRA: Low-Rank Adaptation of Large Language Models (2021) by Hu, Shen, Wallis, Allen-Zhu, Li, L Wang, S Wang, and Chen, https://arxiv.org/abs/2106.09685.

### Pros

- The key functional difference is that our learned weights can be merged with the main weights during
inference, thus not introducing any latency, which is not the case for the adapter layers


## grouped-query attention

## PPO 和 DPO


# 公共数据集
（RefinedWeb、RedPajama、The PILE、Dolma）
