
# Llama Model Implementation

This repository contains an implementation of a Llama-like transformer model architecture, featuring rotary embeddings, multi-head self-attention, MLP layers, and a causal language model head. This architecture is primarily focused on the efficient and scalable use of positional embeddings and attention mechanisms, as used in models like GPT and other transformers. This was created using AutoModelForCausalLM with checkpoint "HuggingFaceTB/SmolLM2-135M".  Below is the model : 
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
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
LlamaForCausalLM                              --
├─LlamaModel: 1-1                             --
│    └─Embedding: 2-1                         28,311,552
│    └─ModuleList: 2-2                        --
│    │    └─LlamaDecoderLayer: 3-1            3,540,096
│    │    └─LlamaDecoderLayer: 3-2            3,540,096
│    │    └─LlamaDecoderLayer: 3-3            3,540,096
│    │    └─LlamaDecoderLayer: 3-4            3,540,096
│    │    └─LlamaDecoderLayer: 3-5            3,540,096
│    │    └─LlamaDecoderLayer: 3-6            3,540,096
│    │    └─LlamaDecoderLayer: 3-7            3,540,096
│    │    └─LlamaDecoderLayer: 3-8            3,540,096
│    │    └─LlamaDecoderLayer: 3-9            3,540,096
│    │    └─LlamaDecoderLayer: 3-10           3,540,096
│    │    └─LlamaDecoderLayer: 3-11           3,540,096
│    │    └─LlamaDecoderLayer: 3-12           3,540,096
│    │    └─LlamaDecoderLayer: 3-13           3,540,096
│    │    └─LlamaDecoderLayer: 3-14           3,540,096
│    │    └─LlamaDecoderLayer: 3-15           3,540,096
│    │    └─LlamaDecoderLayer: 3-16           3,540,096
│    │    └─LlamaDecoderLayer: 3-17           3,540,096
│    │    └─LlamaDecoderLayer: 3-18           3,540,096
│    │    └─LlamaDecoderLayer: 3-19           3,540,096
│    │    └─LlamaDecoderLayer: 3-20           3,540,096
│    │    └─LlamaDecoderLayer: 3-21           3,540,096
│    │    └─LlamaDecoderLayer: 3-22           3,540,096
│    │    └─LlamaDecoderLayer: 3-23           3,540,096
│    │    └─LlamaDecoderLayer: 3-24           3,540,096
│    │    └─LlamaDecoderLayer: 3-25           3,540,096
│    │    └─LlamaDecoderLayer: 3-26           3,540,096
│    │    └─LlamaDecoderLayer: 3-27           3,540,096
│    │    └─LlamaDecoderLayer: 3-28           3,540,096
│    │    └─LlamaDecoderLayer: 3-29           3,540,096
│    │    └─LlamaDecoderLayer: 3-30           3,540,096
│    └─LlamaRMSNorm: 2-3                      576
│    └─LlamaRotaryEmbedding: 2-4              --
├─Linear: 1-2                                 28,311,552
======================================================================
Total params: 162,826,560
Trainable params: 162,826,560
Non-trainable params: 0
======================================================================
```

## Model Architecture

The model is composed of several key components:

- **Rotary Embeddings** (`LlamaRotaryEmbedding`): Efficiently encodes positions using sinusoidal functions, as an alternative to traditional position encodings.
  
- **Self-Attention Mechanism** (`LlamaAttention`): Implements multi-head self-attention with rotary embeddings, using a combination of query, key, and value projections.
  
- **Multilayer Perceptron (MLP)** (`LlamaMLP`): The MLP layer consists of several projections, interspersed with activation functions, used for transforming hidden states.
  
- **Decoder Layer** (`LlamaDecoderLayer`): A core component of the transformer architecture that includes attention and MLP layers with residual connections and layer normalization.

- **Full Transformer Model** (`LlamaModel`): A stack of decoder layers, consisting of the attention and MLP layers. The model also includes token embeddings.

- **Causal Language Model** (`LlamaForCausalLM`): A transformer-based language model designed for autoregressive text generation, with a shared embedding and output layer for more efficient training and inference.

## Installation

To use this model, follow the steps below to set up your environment:

1. Clone the repository:
   ```
   bash
   git clone https://github.com/yourusername/llama-model.git
   cd llama-model
   ```
# Training Logs 
Training logs are there in training.log file. 

# Model Components
-1. LlamaRotaryEmbedding
This module implements the rotary position embeddings, a variant of sinusoidal embeddings, where positions are encoded with sine and cosine functions based on a scaling factor (theta).
```
from model import LlamaRotaryEmbedding

rotary_emb = LlamaRotaryEmbedding(dim=576, theta=10000.0)
input_tensor = torch.randn(2, 10, 576)  # (batch_size, seq_len, hidden_size)
rotary_output = rotary_emb(input_tensor)
print(f"Rotary embedding output shape: {rotary_output.shape}")
```
-2. LlamaAttention
The attention layer implements multi-head self-attention with rotary embeddings. It consists of query, key, and value projections, followed by scaled dot-product attention.
```
from model import LlamaAttention

attention_layer = LlamaAttention(config)
input_tensor = torch.randn(2, 10, 576)  # (batch_size, seq_len, hidden_size)
attention_output = attention_layer(input_tensor)
print(f"Attention output shape: {attention_output.shape}")
```
-3. LlamaMLP
This module defines a feed-forward MLP layer that projects the hidden state to an intermediate size, applies a non-linear activation (SiLU), and then projects it back to the hidden size.
```
from model import LlamaMLP

mlp_layer = LlamaMLP(config)
input_tensor = torch.randn(2, 10, 576)  # (batch_size, seq_len, hidden_size)
mlp_output = mlp_layer(input_tensor)
print(f"MLP output shape: {mlp_output.shape}")

```
-4. LlamaDecoderLayer
This layer combines attention and MLP layers with residual connections, and applies layer normalization before and after attention.

Example Usage:
```
from model import LlamaDecoderLayer

decoder_layer = LlamaDecoderLayer(config)
input_tensor = torch.randn(10, 2, 576)  # (seq_len, batch_size, hidden_size)
decoder_output = decoder_layer(input_tensor)
print(f"Decoder layer output shape: {decoder_output.shape}")

```
-5.LlamaModel
The LlamaModel consists of a stack of decoder layers, which processes input sequences through a series of attention and MLP layers. The model also includes token embeddings for the input sequence.

Example Usage:
```
from model import LlamaModel

model = LlamaModel(config)
input_ids = torch.randint(0, config['vocab_size'], (10, 2))  # (seq_len, batch_size)
model_output = model(input_ids)
print(f"Model output shape: {model_output.shape}")

```
-6. LlamaForCausalLM
This is the final model for causal language modeling. It shares the token embeddings between the input and output layers, making it suitable for text generation tasks.

Example Usage:
```
from model import LlamaForCausalLM

causal_lm_model = LlamaForCausalLM(config)
input_ids = torch.randint(0, config['vocab_size'], (10, 2))  # (seq_len, batch_size)
logits = causal_lm_model(input_ids)
print(f"Logits shape: {logits.shape}")
```

