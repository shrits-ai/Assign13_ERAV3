import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the LlamaAttention mechanism
class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(LlamaAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_size = hidden_size // num_attention_heads
        
        # Linear projections for Q, K, V, and output projection
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.attn_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.attn_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding size
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_attention_heads, self.attn_size).transpose(1, 2)
        k = k.view(B, T, self.num_attention_heads, self.attn_size).transpose(1, 2)
        v = v.view(B, T, self.num_attention_heads, self.attn_size).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.attn_size ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, attn_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        output = self.o_proj(attn_output)
        return output

# Define the LlamaMLP (feed-forward layers) block
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(LlamaMLP, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # Activation function

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        down = self.down_proj(up)
        output = self.act_fn(gate + down)
        return output

# Define RMSNorm for normalization
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LlamaRMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight

# Define the Rotary Embedding class
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super(LlamaRotaryEmbedding, self).__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        
    def forward(self, x):
        # Apply rotary embeddings (simple sine/cosine position encodings for simplicity)
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        # Add rotary embeddings logic as needed here
        return x + position.float()

# Define the LlamaDecoderLayer
class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super(LlamaDecoderLayer, self).__init__()
        self.attn = LlamaAttention(hidden_size, num_attention_heads)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)

    def forward(self, x):
        x_norm = self.input_layernorm(x)
        attn_output = self.attn(x_norm)
        x = x + attn_output  # Residual connection
        x_norm = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output  # Residual connection
        return x

# Define the LlamaForCausalLM Model
class LlamaForCausalLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_layers, intermediate_size, max_position_embeddings):
        super(LlamaForCausalLM, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(hidden_size, num_attention_heads, intermediate_size) for _ in range(num_layers)])
        self.norm = LlamaRMSNorm(hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(max_position_embeddings, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        # Embedding layer
        x = self.embed_tokens(input_ids)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply normalization and rotary embeddings
        x = self.norm(x)
        x = self.rotary_emb(x)
        
        # Final linear layer
        logits = self.lm_head(x)
        return logits

# Model instantiation
model = LlamaForCausalLM(
    vocab_size=49152,
    hidden_size=576,
    num_attention_heads=9,
    num_layers=30,
    intermediate_size=1536,
    max_position_embeddings=2048
)

# Print the model architecture
print(model)
