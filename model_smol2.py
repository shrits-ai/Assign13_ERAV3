import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuration as provided
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
    "rms_norm_eps": 1.0e-05,
    "rope_interleaved": False,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 49152
}

# 1. Rotary Embedding
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Create the position indices
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # Shape: (seq_len, 1)
        freqs = torch.pow(self.theta, -torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)  # Shape: (dim/2,)

        # Reshape freqs for einsum: Shape (dim/2, 1) -> (dim/2, 1) broadcasting with position
        freqs = freqs.unsqueeze(1)  # Shape: (dim/2, 1)

        # Calculate sinusoidal embeddings
        sinusoidal_embeddings = torch.einsum('i,j->ij', position.squeeze(), freqs.squeeze())  # Shape: (seq_len, dim/2)

        # Sinusoidal encoding
        sin = sinusoidal_embeddings.sin().unsqueeze(0)  # Shape: (1, seq_len, dim/2)
        cos = sinusoidal_embeddings.cos().unsqueeze(0)  # Shape: (1, seq_len, dim/2)

        # Concatenate the sin and cos values to create the final embedding
        rotary_embeddings = torch.cat([sin, cos], dim=-1).unsqueeze(0)  # Shape: (1, seq_len, dim)

        # Remove the extra leading dimension (1) to match input tensor shape
        return rotary_embeddings.squeeze(0)  # Shape: (seq_len, dim)
'''
# Testing LlamaRotaryEmbedding again with the modified code
rotary_emb = LlamaRotaryEmbedding(dim=576, theta=10000.0)
input_tensor = torch.randn(2, 10, 576)  # (batch_size, seq_len, hidden_size)
rotary_output = rotary_emb(input_tensor)
print(f"Rotary embedding output shape: {rotary_output.shape}")
'''


# 2. Attention Layer
class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
        self.k_proj = nn.Linear(config['hidden_size'], config['hidden_size'] // 3, bias=False)
        self.v_proj = nn.Linear(config['hidden_size'], config['hidden_size'] // 3, bias=False)
        self.o_proj = nn.Linear(config['hidden_size'] // 3, config['hidden_size'], bias=False)  # Adjust output projection size
        self.rope_emb = LlamaRotaryEmbedding(config['hidden_size'])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()  # Get the batch size and sequence length
        q = self.q_proj(x)  # Shape: (batch_size, seq_len, hidden_size)
        k = self.k_proj(x)  # Shape: (batch_size, seq_len, hidden_size // 3)
        v = self.v_proj(x)  # Shape: (batch_size, seq_len, hidden_size // 3)

        # Apply rotary embeddings (positional encoding)
        q, k = self.rope_emb(q), self.rope_emb(k)

        # Calculate attention weights (scaled dot-product attention)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, seq_len, seq_len)
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)  # Shape: (batch_size, seq_len, seq_len)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # Shape: (batch_size, seq_len, hidden_size // 3)

        # Output projection (adjusted to match hidden_size)
        out = self.o_proj(attn_output)  # Shape: (batch_size, seq_len, hidden_size)

        return out
'''
# Testing LlamaAttention again
attention_layer = LlamaAttention(config)
input_tensor = torch.randn(2, 10, 576)  # (batch_size, seq_len, hidden_size)
attention_output = attention_layer(input_tensor)
print(f"Attention output shape: {attention_output.shape}")
'''

# 3. MLP Layer
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)  # Hidden size to intermediate size
        self.up_proj = nn.Linear(config['intermediate_size'], config['intermediate_size'], bias=False)  # Intermediate size to intermediate size
        self.down_proj = nn.Linear(config['intermediate_size'], config['hidden_size'], bias=False)  # Intermediate size to hidden size
        self.act_fn = torch.nn.SiLU()  # Activation function

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Flatten input to (batch_size * seq_len, hidden_size) for projection
        x = x.view(batch_size * seq_len, -1)  # Shape: (batch_size * seq_len, hidden_size)

        # Apply gate projection
        x = self.gate_proj(x)  # Shape: (batch_size * seq_len, intermediate_size)
        x = self.act_fn(x)  # Apply activation

        # Apply up projection
        x = self.up_proj(x)  # Shape: (batch_size * seq_len, intermediate_size)

        # Apply down projection
        x = self.down_proj(x)  # Shape: (batch_size * seq_len, hidden_size)

        # Reshape back to (batch_size, seq_len, hidden_size)
        x = x.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, hidden_size)

        return x
'''
# Test the MLP again
mlp_layer = LlamaMLP(config)
input_tensor = torch.randn(2, 10, 576)  # (batch_size, seq_len, hidden_size)
mlp_output = mlp_layer(input_tensor)
print(f"MLP output shape: {mlp_output.shape}")
'''


# 4. Decoder Layer
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.LayerNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = nn.LayerNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, x):
        # Apply input normalization
        x = self.input_layernorm(x)
        
        # Attention
        attn_output = self.self_attn(x)
        x = x + attn_output  # Residual connection
        
        # Apply post-attention layer normalization
        x = self.post_attention_layernorm(x)

        # Apply MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Residual connection
        return x
'''
# Testing LlamaDecoderLayer
decoder_layer = LlamaDecoderLayer(config)
input_tensor = torch.randn(10, 2, 576)  # (seq_len, batch_size, hidden_size)
decoder_output = decoder_layer(input_tensor)
print(f"Decoder layer output shape: {decoder_output.shape}")

# 5. Model
class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])

        # Partially shared decoder layers
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config['num_hidden_layers'])])

        # Separate adapters for each layer (adds more parameters)
        self.adapters = nn.ModuleList([
            nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
            for _ in range(config['num_hidden_layers'])
        ])

        self.norm = nn.LayerNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(self, input_ids):
        # Initial embedding lookup
        x = self.embed_tokens(input_ids)

        # Pass through transformer layers with unique adapters per layer
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Apply the i-th decoder layer
            x = x + self.adapters[i](x)  # Add per-layer adapter

        # Apply the final layer normalization
        x = self.norm(x)
        return x


'''
class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config['num_hidden_layers'])])
        self.norm = nn.LayerNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.rotary_emb = LlamaRotaryEmbedding(config['hidden_size'])

    def forward(self, input_ids):
        # Initial embedding lookup
        x = self.embed_tokens(input_ids)

        # Pass through the transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply the final layer normalization
        x = self.norm(x)
        return x
'''
# Testing LlamaModel
model = LlamaModel(config)
input_ids = torch.randint(0, config['vocab_size'], (10, 2))  # (seq_len, batch_size)
model_output = model(input_ids)
print(f"Model output shape: {model_output.shape}")
'''
# 6. Causal Language Model (Final Model)
class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        # Share weights between the embedding and output layers
        #self.lm_head = self.model.embed_tokens

        self.lm_head= nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

# Testing LlamaForCausalLM
'''
causal_lm_model = LlamaForCausalLM(config_model)
print(causal_lm_model)
from torchinfo import summary
summary( causal_lm_model )
input_ids = torch.randint(0, config_model['vocab_size'], (10, 2))  # (seq_len, batch_size)
logits = causal_lm_model(input_ids)
print(f"Logits shape: {logits.shape}")
'''


