import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# Factorized Linear Layer
class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.low_rank_proj = nn.Linear(in_features, rank, bias=False)
        self.high_rank_proj = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        # Low-rank approximation: W = U * V, where U and V are learned matrices
        return self.high_rank_proj(self.low_rank_proj(x))


# Rotary Embedding
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        freqs = torch.pow(self.theta, -torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        sinusoidal_embeddings = torch.einsum('i,j->ij', position.squeeze(), freqs.squeeze())
        sin = sinusoidal_embeddings.sin().unsqueeze(0)
        cos = sinusoidal_embeddings.cos().unsqueeze(0)
        return torch.cat([sin, cos], dim=-1).squeeze(0)


# Attention Layer with Factorized Linear
class LlamaAttention(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        # Use Xavier or He initialization for better gradient flow
        self.q_proj = FactorizedLinear(config['hidden_size'], config['hidden_size'], rank)
        self.k_proj = FactorizedLinear(config['hidden_size'], config['hidden_size'] // 3, rank)
        self.v_proj = FactorizedLinear(config['hidden_size'], config['hidden_size'] // 3, rank)
        self.o_proj = FactorizedLinear(config['hidden_size'] // 3, config['hidden_size'], rank)
        self.rope_emb = LlamaRotaryEmbedding(config['hidden_size'])
        self.act_fn = nn.GELU()  # Switching to GELU for better gradient flow

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k = self.rope_emb(q), self.rope_emb(k)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        return self.o_proj(attn_output)



# MLP Layer with Factorized Linear
class LlamaMLP(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.gate_proj = FactorizedLinear(config['hidden_size'], config['intermediate_size'], rank)
        self.up_proj = FactorizedLinear(config['intermediate_size'], config['intermediate_size'], rank)
        self.down_proj = FactorizedLinear(config['intermediate_size'], config['hidden_size'], rank)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        x = self.gate_proj(x)
        x = self.act_fn(x)
        x = self.up_proj(x)
        return self.down_proj(x)


# Decoder Layer
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.self_attn = LlamaAttention(config, rank)
        self.mlp = LlamaMLP(config, rank)
        self.input_layernorm = nn.LayerNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = nn.LayerNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# Llama Model
class LlamaModel(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, rank) for _ in range(config['num_hidden_layers'])])
        self.norm = nn.LayerNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# Causal Language Model with Tied Embeddings
class LlamaForCausalLM(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.model = LlamaModel(config, rank)
        # Tie embeddings: Share input and output embeddings
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits



# Test the updated model
config_model = {
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "silu",
    "hidden_size": 576,
    "initializer_range": 0.041666666666666664,
    "intermediate_size": 1536,
    "is_llama_config": True,
    "max_position_embeddings": 2048,
    "num_attention_heads": 9,
    "num_hidden_layers": 30,
    "num_key_value_heads": 3,
    "pad_token_id": None,
    "pretraining_tp": 1,
    "rms_norm_eps": 1.0e-06,
    "rope_interleaved": False,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 49152
}
#Testing
rank = 242  # Rank for low-rank approximations
causal_lm_model = LlamaForCausalLM(config_model, rank)
input_ids = torch.randint(0, config_model['vocab_size'], (1, 128))
#input_ids = torch.randint(0, config_model['vocab_size'], (10, 2))  # (seq_len, batch_size)
logits = causal_lm_model(input_ids)
print(f"Logits shape: {logits.shape}")
print(causal_lm_model)
summary(causal_lm_model)
print(f"Logits Shape: {logits.shape}")  # Should be (batch_size, seq_len, vocab_size)
print(f"Logits (Last Token): {logits[0, -1, :5]}") 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
print(len(tokenizer))
input_ids = tokenizer.encode("Gravity is", return_tensors="pt").to("cpu")
outputs = causal_lm_model(input_ids)
logits = outputs[:, -1, :]  # Last token's logits
next_token_id = torch.argmax(logits, dim=-1).item()
print(f"Next token ID: {next_token_id}")
print(f"Next token: {tokenizer.decode([next_token_id])}")
 # First 5 logits of the last token
input_ids = tokenizer.encode("Gravity is", return_tensors="pt").to("cpu")
for _ in range(50):  # Generate up to 50 tokens
    outputs = causal_lm_model(input_ids)
    logits = outputs[:, -1, :]  # Last token logits
    next_token_id = torch.argmax(logits, dim=-1).item()  # Select most likely token
    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to("cpu")], dim=1)
    if next_token_id == tokenizer.eos_token_id:  # Stop if EOS is generated
        break

generated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
print(f"Generated Text: {generated_text}")
'''
# Configuration parameters
vocab_size = 49152
hidden_size = 576
intermediate_size = 1536
num_attention_heads = 9
num_hidden_layers = 30
rank = 242

# 1. Embedding layer
embed_tokens_params = vocab_size * hidden_size

# 2. Attention layer parameters (4 FactorizedLinear layers)
def factorized_linear_params(in_features, out_features, rank):
    return (in_features * rank) + (rank * out_features)

# q_proj, k_proj, v_proj, o_proj parameters
q_proj_params = factorized_linear_params(hidden_size, hidden_size, rank)
k_proj_params = factorized_linear_params(hidden_size, hidden_size // 3, rank)
v_proj_params = factorized_linear_params(hidden_size, hidden_size // 3, rank)
o_proj_params = factorized_linear_params(hidden_size // 3, hidden_size, rank)

# Total attention layer parameters (q_proj + k_proj + v_proj + o_proj)
attention_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params

# 3. MLP layer parameters (3 FactorizedLinear layers)
gate_proj_params = factorized_linear_params(hidden_size, intermediate_size, rank)
up_proj_params = factorized_linear_params(intermediate_size, intermediate_size, rank)
down_proj_params = factorized_linear_params(intermediate_size, hidden_size, rank)

# Total MLP parameters (gate_proj + up_proj + down_proj)
mlp_params = gate_proj_params + up_proj_params + down_proj_params

# 4. Decoder layer parameters (self-attention + MLP + LayerNorms)
decoder_layer_params = attention_params + mlp_params + 2 * hidden_size  # +2 for the LayerNorms

# 5. Total parameters for all decoder layers
total_decoder_layers_params = num_hidden_layers * decoder_layer_params

# 6. Output layer parameters (lm_head)
lm_head_params = hidden_size * vocab_size

# 7. Total parameters in the model
total_params = embed_tokens_params + total_decoder_layers_params + lm_head_params

print(f"Total parameters: {total_params}")

'''
