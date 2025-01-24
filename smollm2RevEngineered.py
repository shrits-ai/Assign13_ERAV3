import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pdb

def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Custom model architecture
class LlamaAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, kv_size):
        super().__init__()
        self.num_heads = num_heads
        self.kv_size = kv_size
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Projections for query, key, and value
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, kv_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, kv_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        pdb.set_trace()
        B, T = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)
        C = self.hidden_size

        # Project inputs to query, key, and value
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.kv_size // self.num_heads).transpose(1, 2)  # (B, nh, T, kv_dim/nh)
        v = self.v_proj(x).view(B, T, self.num_heads, self.kv_size // self.num_heads).transpose(1, 2)  # (B, nh, T, kv_dim/nh)

        # Compute scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.kv_size // self.num_heads))
        att = F.softmax(att, dim=-1)  # Normalize attention scores
        y = att @ v  # Attention output: (B, nh, T, T) x (B, nh, T, kv_dim/nh) -> (B, nh, T, kv_dim/nh)

        # Reshape output back to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply final output projection
        y = self.o_proj(y)

        return y


class LlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) + self.up_proj(hidden_states))

class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return hidden_states / (hidden_states.norm(2, dim=-1, keepdim=True) + self.eps) * self.weight

class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, kv_size, eps):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads, kv_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps)

    def forward(self, hidden_states):
        normed_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(normed_states)
        attention_output += hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(attention_output))
        return mlp_output + attention_output

class LlamaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config["model"]["model_config"]
        self.embed_tokens = torch.nn.Embedding(model_config["vocab_size"], model_config["hidden_size"])
        self.layers = torch.nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=model_config["hidden_size"],
                intermediate_size=model_config["intermediate_size"],
                num_heads=model_config["num_attention_heads"],
                kv_size=model_config["hidden_size"] // model_config["num_key_value_heads"],
                eps=model_config["rms_norm_eps"],
            ) for _ in range(model_config["num_hidden_layers"])
        ])
        self.norm = LlamaRMSNorm(model_config["hidden_size"], model_config["rms_norm_eps"])

    def forward(self, input_ids):
        # Concatenate the list of tensors into a single tensor
        input_ids = torch.cat(input_ids, dim=0) if isinstance(input_ids, list) else input_ids

        # Get the device from the first model parameter if input_ids is a tensor
        device = input_ids.device if isinstance(input_ids, torch.Tensor) else next(self.parameters()).device

        # Move the input_ids tensor to the correct device
        input_ids = input_ids.to(device)
        # Apply embedding to input_ids (this creates a 3D tensor: batch_size, sequence_length, hidden_size)
        # Apply embedding to input_ids
        hidden_states = self.embed_tokens(input_ids)

        # Debug the shape of hidden_states after embedding
        print(f"Shape after embedding: {hidden_states.shape}")

        # Apply layers (e.g., self-attention, etc.)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Debug the shape of hidden_states after all layers
        print(f"Shape after layers: {hidden_states.shape}")

        # Apply normalization
        normed_states = self.input_layernorm(hidden_states)
        print(f"Shape after layernorm: {normed_states.shape}")

        # The problematic line
        return normed_states / (normed_states.norm(2, dim=-1, keepdim=True) + self.eps) * self.weight


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = LlamaModel(config)

        self.lm_head = torch.nn.Linear(
            config["model"]["model_config"]["hidden_size"],
            config["model"]["model_config"]["vocab_size"],
            bias=False
        )

    def forward(self, input_ids):
        self.model.to(self.device)
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)

# PyTorch Lightning module
class TransformerLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(TransformerLightningModule, self).__init__()
        self.save_hyperparameters(config)

        self.model = LlamaForCausalLM(config)

        self.learning_rate = config["optimizer"]["learning_rate_scheduler"]["learning_rate"]
        self.weight_decay = config["optimizer"]["weight_decay"]
        self.clip_grad = config["optimizer"]["clip_grad"]

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        logits = self(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def lr_lambda(self, current_step):
        config = self.hparams["optimizer"]["learning_rate_scheduler"]
        warmup_steps = config["lr_warmup_steps"]
        decay_steps = config["lr_decay_steps"]

        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)

        return max(0.0, 1 - (current_step - warmup_steps) / float(decay_steps))

# DataModule
class TextDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name_or_path"])
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        self.batch_size = self.config["tokens"]["micro_batch_size"]
        self.num_workers = self.config["data_stages"][0]["data"]["num_loading_workers"]

    def setup(self, stage=None):
        data_config = self.config["data_stages"][0]["data"]
        data_set = data_config["dataset"]["dataset_folder"][0]
        
        # Using `streaming=True`, hence, avoid directly indexing
        self.dataset = load_dataset(
            data_set,
            'cosmopedia-v2',
            streaming=True,
        )

        # Tokenize the dataset using `map`
        self.dataset = self.dataset.map(
            self.tokenize_text,
            batched=True,
            remove_columns=["text"]
        )


    def tokenize_text(self, x):
        encodings = self.tokenizer(
            x["text"],
            truncation=True,
            max_length=self.config["tokens"]["sequence_length"],
            padding="max_length",
        )
        return {"input_ids": encodings["input_ids"]}

    def train_dataloader(self):
        train_data = self.dataset["train"]
        # Using `iter()` to stream batches for the dataset
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


from pytorch_lightning.callbacks import ModelCheckpoint

# Training script
def train(config_path):
    config = load_yaml_config(config_path)
    model = TransformerLightningModule(config)
    data_module = TextDataModule(config)
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        max_steps=config["tokens"]["train_steps"],
        accelerator="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        devices=1,
        callbacks=[checkpoint_callback],
        precision=16 if torch.cuda.is_available() else 32,
    )
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    train("config.yaml")
