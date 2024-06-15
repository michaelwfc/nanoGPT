
# Mixtral

[Mixtral of Experts](https://arxiv.org/abs/2401.04088)
-  Mixtral 8x7, Mixtral 8x22B 

## Mixtral of Experts（MoE）模型

MoE 是混合专家（Mixture of Experts）的缩写，这是一类将多个较小「专家」子网络组合起来得到的集成模型。每个子网络都负责处理不同类型的任务。通过使用多个较小的子网络，而不是一个大型网络，MoE 可以更高效地分配计算资源。这让它们可以更有效地扩展，并可望在更广泛的任务上实现更好的性能。

## Mixtral 架构

关键思路是将 Transformer 架构中的每个前馈模块替换成 8 个专家层

# [Llama 3](https://ai.meta.com/blog/meta-llama-3/)

- Llama 3 8B 可能最能吸引各种微调用户，因为使用 LoRA 在单台 GPU 上就能轻松对其进行微调。


## Phi-3

- 微软 Phi-3 3.8B 可能比较适合用于移动设备；其作者表示，Phi-3 3.8B 的一个量化版本可以运行在 iPhone 14 上

## TinyLlama 
[TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385)
- https://github.com/jzhang38/TinyLlama
- 


