## Custom LLM Training Framework
A minimal implementation for training medium-sized language models with efficient attention mechanisms, compatible with Apple Silicon (MPS) and CUDA.This was created using AutoModelForCausalLM with checkpoint "HuggingFaceTB/SmolLM2-135M".  
Below is the reference model : 
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
Model parameters: 162M (approx)
```
# Model Architecture (model.py)
```
Model parameters: 134.52M
CustomLLM(
  (embed_tokens): Embedding(49152, 576)
  (layers): ModuleList(
    (0-29): 30 x DecoderLayer(
      (self_attn): CustomAttention(
        (q_proj): Linear(in_features=576, out_features=576, bias=False)
        (k_proj): Linear(in_features=576, out_features=192, bias=False)
        (v_proj): Linear(in_features=576, out_features=192, bias=False)
        (o_proj): Linear(in_features=576, out_features=576, bias=False)
        (rotary_emb): RotaryEmbedding()
      )
      (mlp): CustomMLP(
        (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
        (up_proj): Linear(in_features=576, out_features=1536, bias=False)
        (down_proj): Linear(in_features=1536, out_features=576, bias=False)
        (act_fn): SiLU()
      )
      (input_norm): CustomRMSNorm()
      (post_attn_norm): CustomRMSNorm()
    )
  )
  (norm): CustomRMSNorm()
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
```

## Key Specifications:
```
- Parameters: ~135M (configurable)
- Hidden Size: 576
- Layers: 30
- Attention Heads: 9 (Query), 3 (Key/Value)
- Sequence Length: 2048 (max)
- Vocabulary Size: 49,152
- Rotary Positional Embeddings (θ=10000)
- RMSNorm for Layer Normalization
```
## Architectural Features
1. Memory-Efficient Attention:
```
- Grouped Query Attention (GQA) reduces KV-heads
- Rotary Position Embeddings (RoPE)
- Causal attention mask with padding support
```
2. Custom Components:
```
- Sliding Window Attention (future extension)
- SiLU-activated MLP (FFN) with gating
- Weight-Tied Embeddings (input/output)
- Gradient Checkpointing Ready
```
3. Optimization:
```
- MPS/CPU/CUDA compatible
- 16-bit precision support (disabled for MPS)
- KV-caching in generation
```
# Training Setup (train.py)
## Core Training Configuration:
```
- Batch Size: 4 (effective 32 with gradient accumulation)
- Context Length: 256 tokens
- Optimizer: AdamW (lr=2e-4, weight_decay=0.01)
- Dataset: cosmopedia-v2 (streaming)
- Training Steps: 5000 + 50 (phased)
```
## Key Implementation Details
1. Efficient Data Handling:
```
- StreamingDataset for large corpora
- Dynamic batching with padding
- On-the-fly tokenization
- DataCollatorForLanguageModeling (HF)
```
2. Training Infrastructure:
```
- Accelerate Library Integration
- Gradient Accumulation (8 steps)
- Mixed Precision Training
- Automatic Checkpointing
- W&B Logging Integration
```
3. Special Features:
```
- MPS Memory Management Callback
- Text Generation Progress Monitoring
- Phase-Based Training (warmup + main)
- Model Parallelism Ready
```

# Training Logs 
Training logs are there in training.log file. Retraining started from 5001. Log says step 5000 because step starts from 0. 
```
No checkpoint found, starting training from scratch...
{'loss': 9.7599, 'grad_norm': 1.6781054735183716, 'learning_rate': 4e-05, 'epoch': 0.02}                                                                     
{'loss': 7.6304, 'grad_norm': 1.1875282526016235, 'learning_rate': 8e-05, 'epoch': 0.04}                                                                     
{'loss': 6.5971, 'grad_norm': 1.3352041244506836, 'learning_rate': 0.00012, 'epoch': 0.06}                                                                   
{'loss': 5.9583, 'grad_norm': 1.6744577884674072, 'learning_rate': 0.00016, 'epoch': 0.08}                                                                   
Generated text at step 500:
--------------------------------------------------
The future of AI is a fascinating way of the world. This is a type of learning that has been a significant role in the United. This process is known as the "The World of the 190, and the 19th century. This is the concept of the United States, which is a crucial role in the United States.

In the context of the United, it is essential to understand the concept of the early 1900. This is the concept of the United States,
--------------------------------------------------
{'loss': 5.4655, 'grad_norm': 1.2713903188705444, 'learning_rate': 0.0002, 'epoch': 0.1}                                                                     
{'eval_loss': 5.2323503494262695, 'eval_runtime': 12.2146, 'eval_samples_per_second': 8.187, 'eval_steps_per_second': 1.064, 'epoch': 0.1}       
{'loss': 5.0969, 'grad_norm': 1.2072104215621948, 'learning_rate': 0.00019555555555555556, 'epoch': 0.12}                                                    
{'loss': 4.8178, 'grad_norm': 1.0987201929092407, 'learning_rate': 0.00019111111111111114, 'epoch': 0.14}                                                    
{'loss': 4.6386, 'grad_norm': 1.1196931600570679, 'learning_rate': 0.0001866666666666667, 'epoch': 0.16}                                                     
{'loss': 4.4664, 'grad_norm': 1.087496042251587, 'learning_rate': 0.00018222222222222224, 'epoch': 0.18}                                                     
Generated text at step 1000:
--------------------------------------------------
The future of AI is a crucial aspect of the world's health and wellbeing. It is a crucial aspect of the business and its environment, including the environment, the government, and the environment. In this course unit, we will explore the concept of the United States, its history, and how it relates to the world around them. We will also examine how these factors can help us understand the world around us and how they can help us understand the world around us.

To begin, let's understand what we mean
--------------------------------------------------
{'loss': 4.3394, 'grad_norm': 1.0962802171707153, 'learning_rate': 0.00017777777777777779, 'epoch': 0.2}                                                     
{'eval_loss': 4.249337673187256, 'eval_runtime': 16.183, 'eval_samples_per_second': 6.179, 'eval_steps_per_second': 0.803, 'epoch': 0.2}  
{'loss': 4.243, 'grad_norm': 0.9174259901046753, 'learning_rate': 0.00017333333333333334, 'epoch': 0.22}                                                     
{'loss': 4.1349, 'grad_norm': 0.9104893207550049, 'learning_rate': 0.00016888888888888889, 'epoch': 0.24}                                                    
{'loss': 4.0304, 'grad_norm': 1.0610218048095703, 'learning_rate': 0.00016444444444444444, 'epoch': 0.26}                                                    
{'loss': 3.9556, 'grad_norm': 0.9328898191452026, 'learning_rate': 0.00016, 'epoch': 0.28}                                                                   
Generated text at step 1500:
--------------------------------------------------
The future of AI is a crucial aspect of human behavior and development. It involves using the process of creating and managing data, which can be used to create new ways of thinking and learning. In this chapter, we will explore the concept of data analysis in the context of the internet, specifically focusing on the concept of data analysis. We will also examine how to analyze data using Python, which is a powerful tool for understanding and understanding the role of data analysis in the field of data science.

To begin, let's
--------------------------------------------------
{'loss': 3.8891, 'grad_norm': 1.0196585655212402, 'learning_rate': 0.00015555555555555556, 'epoch': 0.3}  
{'eval_loss': 3.821467399597168, 'eval_runtime': 75.9033, 'eval_samples_per_second': 1.317, 'eval_steps_per_second': 0.171, 'epoch': 0.3}
{'loss': 3.8285, 'grad_norm': 0.9595060348510742, 'learning_rate': 0.0001511111111111111, 'epoch': 0.32}                                                     
{'loss': 3.7686, 'grad_norm': 1.0341150760650635, 'learning_rate': 0.00014666666666666666, 'epoch': 0.34}                                                    
{'loss': 3.7121, 'grad_norm': 1.0021989345550537, 'learning_rate': 0.00014222222222222224, 'epoch': 0.36}    
{'loss': 3.6646, 'grad_norm': 0.9691662192344666, 'learning_rate': 0.0001377777777777778, 'epoch': 0.38}   
Generated text at step 2000:
--------------------------------------------------
The future of AI is an exciting and rewarding way to do just that! It's called a "p-rays," which are like a big building where people can learn about different things.

Now, imagine if you could build a treehouse with your friends and family, but instead of using your toys, you could use your computer to communicate and communicate with each other. That's what a "p-ray" is! It's a way of making sure that every computer is connected and connected to the internet, so
--------------------------------------------------
{'loss': 3.6285, 'grad_norm': 0.8970617055892944, 'learning_rate': 0.00013333333333333334, 'epoch': 0.4}                                                     
{'eval_loss': 3.5742218494415283, 'eval_runtime': 13.8225, 'eval_samples_per_second': 7.235, 'eval_steps_per_second': 0.94, 'epoch': 0.4}  
{'loss': 3.5743, 'grad_norm': 0.9362289905548096, 'learning_rate': 0.00012888888888888892, 'epoch': 0.42}                                                    
{'loss': 3.5363, 'grad_norm': 0.955926775932312, 'learning_rate': 0.00012444444444444444, 'epoch': 0.44}    
{'loss': 3.5186, 'grad_norm': 0.924082338809967, 'learning_rate': 0.00012, 'epoch': 0.46}                                                                 
{'loss': 3.4745, 'grad_norm': 0.97138512134552, 'learning_rate': 0.00011555555555555555, 'epoch': 0.48}   
Generated text at step 2500:
--------------------------------------------------
The future of AI is a powerful tool that allows individuals to create and manage their own tasks effectively. One such tool is the C&D, which is a type of software that allows users to create and manage their tasks efficiently. In this section, we will explore how to create a simple software that allows users to create a more efficient and efficient software that can be used to create a more efficient and efficient software.

First, let's talk about what a software is. A software is a collection of software that allows
--------------------------------------------------
{'loss': 3.4505, 'grad_norm': 1.0667750835418701, 'learning_rate': 0.00011111111111111112, 'epoch': 0.5}                                                     
{'eval_loss': 3.39498233795166, 'eval_runtime': 13.5543, 'eval_samples_per_second': 7.378, 'eval_steps_per_second': 0.959, 'epoch': 0.5}   
{'loss': 3.408, 'grad_norm': 0.9481505155563354, 'learning_rate': 0.00010666666666666667, 'epoch': 0.52}                                                     
{'loss': 3.375, 'grad_norm': 0.9432135224342346, 'learning_rate': 0.00010222222222222222, 'epoch': 0.54}                                                     
{'loss': 3.3449, 'grad_norm': 0.9301525950431824, 'learning_rate': 9.777777777777778e-05, 'epoch': 0.56}                                                     
{'loss': 3.3049, 'grad_norm': 1.0105029344558716, 'learning_rate': 9.333333333333334e-05, 'epoch': 0.58}     
Generated text at step 3000:
--------------------------------------------------
The future of AI is a complex and multifaceted field that combines elements of science, technology, and computer science. One of the most important aspects of this field is the use of technology to enhance and improve the quality of life for the future. In this section, we will explore how technology can be used to improve the quality of life for young people.

First, let's define what we mean by "digital" and "digital. Digital refers to the ability to communicate, communicate, and interact with one another in a
--------------------------------------------------
{'loss': 3.3081, 'grad_norm': 0.9850307703018188, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.6}  
{'eval_loss': 3.2645599842071533, 'eval_runtime': 12.6764, 'eval_samples_per_second': 7.889, 'eval_steps_per_second': 1.026, 'epoch': 0.6}  
{'loss': 3.2501, 'grad_norm': 1.0114974975585938, 'learning_rate': 8e-05, 'epoch': 0.64}                                                                     
{'loss': 3.2203, 'grad_norm': 1.0029985904693604, 'learning_rate': 7.555555555555556e-05, 'epoch': 0.66}                                                     
{'loss': 3.2146, 'grad_norm': 0.9595005512237549, 'learning_rate': 7.111111111111112e-05, 'epoch': 0.68}       
Generated text at step 3500:
--------------------------------------------------
The future of AI is an essential aspect of human communication and human interaction. This tutorial will guide you through the process of using AI in a computer, providing in-depth explanations and key tips along the way.

**Step 1: Understand What AI Is**
Before diving into the specifics of AI, let's start with the basics. AI is a computer-like machine that can perform tasks, such as memory, memory, and communication. It's a computer-generated system that allows humans to interact with computers
--------------------------------------------------
{'loss': 3.1853, 'grad_norm': 0.9829711318016052, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.7}                                                      
{'eval_loss': 3.154695987701416, 'eval_runtime': 14.4317, 'eval_samples_per_second': 6.929, 'eval_steps_per_second': 0.901, 'epoch': 0.7}  
{'loss': 3.1776, 'grad_norm': 1.0177723169326782, 'learning_rate': 6.222222222222222e-05, 'epoch': 0.72}                                                     
{'loss': 3.1496, 'grad_norm': 0.9958975911140442, 'learning_rate': 5.7777777777777776e-05, 'epoch': 0.74}                                                    
{'loss': 3.1408, 'grad_norm': 1.0646604299545288, 'learning_rate': 5.333333333333333e-05, 'epoch': 0.76}                                                     
{'loss': 3.1363, 'grad_norm': 1.0250983238220215, 'learning_rate': 4.888888888888889e-05, 'epoch': 0.78}     
Generated text at step 4000:
--------------------------------------------------
The future of AI is a fascinating field that combines the principles of artificial intelligence (AI) and artificial intelligence (AI) to create intelligent machines that can solve complex problems. One such area is the use of AI in AI and AI, particularly in the context of AI and AI. In this section, we will explore how AI can be used to improve AI and its applications in AI and AI.

Before we dive into the details, let's first understand what AI is. AI is a type of artificial intelligence that uses
--------------------------------------------------
{'loss': 3.1013, 'grad_norm': 0.9849228262901306, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.8}  
{'eval_loss': 3.081460952758789, 'eval_runtime': 65.3653, 'eval_samples_per_second': 1.53, 'eval_steps_per_second': 0.199, 'epoch': 0.8} 
{'loss': 3.0913, 'grad_norm': 0.9926738739013672, 'learning_rate': 4e-05, 'epoch': 0.82}                                                                     
{'loss': 3.0809, 'grad_norm': 0.954986572265625, 'learning_rate': 3.555555555555556e-05, 'epoch': 0.84}    
{'loss': 3.0732, 'grad_norm': 1.0269407033920288, 'learning_rate': 3.111111111111111e-05, 'epoch': 0.86}                                                     
{'loss': 3.0404, 'grad_norm': 1.0549601316452026, 'learning_rate': 2.6666666666666667e-05, 'epoch': 0.88}    
Generated text at step 4500:
--------------------------------------------------
The future of AI is a fascinating area of study that combines the principles of artificial intelligence (AI) and artificial intelligence (AI) to create new technologies and technologies. This field combines elements of science, engineering, and computer science to create innovative solutions for various industries, including healthcare. In this course unit, we will explore how AI can be used to improve healthcare and healthcare for individuals with disabilities.

First, let's define what we mean by AI. AI refers to the ability of computers to learn and understand, rather
--------------------------------------------------
{'loss': 3.0462, 'grad_norm': 1.0019572973251343, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.9}   
{'eval_loss': 3.0247130393981934, 'eval_runtime': 57.5065, 'eval_samples_per_second': 1.739, 'eval_steps_per_second': 0.226, 'epoch': 0.9}   
{'loss': 3.0387, 'grad_norm': 1.0226173400878906, 'learning_rate': 1.777777777777778e-05, 'epoch': 0.92}    
{'loss': 3.0278, 'grad_norm': 1.0116548538208008, 'learning_rate': 1.3333333333333333e-05, 'epoch': 0.94}  
{'loss': 3.0159, 'grad_norm': 1.0547219514846802, 'learning_rate': 8.88888888888889e-06, 'epoch': 0.96}   
{'loss': 3.0213, 'grad_norm': 0.9895848035812378, 'learning_rate': 4.444444444444445e-06, 'epoch': 0.98}   
Generated text at step 5000:
--------------------------------------------------
The future of AI is a fascinating field that combines the power of technology to create new technologies and technologies. One such technology is the use of artificial intelligence (AI) in the context of artificial intelligence (AI), which has revolutionized the way we communicate and interact with the world. In this section, we will explore how AI can be used to analyze and analyze AI in the context of AI and its applications.

First, let's define what AI is. AI is a type of AI that uses computers to learn and learn
--------------------------------------------------
{'loss': 3.0184, 'grad_norm': 1.0157222747802734, 'learning_rate': 0.0, 'epoch': 1.0}                                                                        
{'eval_loss': 2.996417284011841, 'eval_runtime': 12.7959, 'eval_samples_per_second': 7.815, 'eval_steps_per_second': 1.016, 'epoch': 1.0}                    
{'train_runtime': 20461.8487, 'train_samples_per_second': 7.819, 'train_steps_per_second': 0.244, 'train_loss': 3.884700964355469, 'epoch': 1.0}  


Found checkpoint at ./checkpoints/checkpoint-5000, resuming training... logging interval changed to 10

```

# Usage:
Training Command:
```
# Phase 1: Initial Training
python train.py --phase init

# Phase 2: Resume Training
python train.py --phase resume --checkpoint ./checkpoints/final_5000
```

# Huggingface app space detail: 
```
https://huggingface.co/spaces/Shriti09/Smol2TextGenerator
```
