import os
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from time import time
from pytorch_lightning.loggers import CSVLogger

# Check available device (CUDA, MPS, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define the streaming dataset class
class StreamingDataset(Dataset):
    def __init__(self, dataset_name, tokenizer_name, batch_size=8, max_len=2048):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        
        # Load dataset with streaming option
        self.dataset = load_dataset(dataset_name, 'cosmopedia-v2', split='train', streaming=True)
        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = self.tokenizer.vocab_size  # Use tokenizer's vocab size

    def __len__(self):
        # Return an arbitrary large number as we are streaming the dataset
        return 1000  # This is just a placeholder; streaming datasets don't need a fixed size

    def __getitem__(self, idx):
        # Fetch the next sample from the dataset
        sample = next(iter(self.dataset))  # Get the next sample from the stream
        text = sample['text']  # Assuming the 'text' field contains the raw text
        #print(f"Fetched text: {text[:100]}...")
        # Tokenize the text and convert it to input IDs
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)  # Convert from (1, max_len) to (max_len,)
        
        return input_ids

# Define the Llama model using PyTorch Lightning
class LlamaForCausalLM(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_layers, intermediate_size, max_position_embeddings):
        super(LlamaForCausalLM, self).__init__()

        self.save_hyperparameters()

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(hidden_size, num_attention_heads, intermediate_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def training_step(self, batch, batch_idx):
        start_time = time()  # Track batch processing time
        input_ids = batch.to(self.device)

        # Forward pass
        output = self(input_ids)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), input_ids.view(-1))
        # Log the loss and time taken for the batch
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("batch_time", time() - start_time, on_step=True, on_epoch=False, prog_bar=True)

        # Debug log to check how long the training step takes
        #print(f"Batch {batch_idx}: Loss: {loss.item()}, Time: {time() - start_time:.2f} seconds")
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.003)
        return optimizer

    def train_dataloader(self):
        # Use the StreamingDataset class
        train_dataset = StreamingDataset(
            dataset_name="HuggingFaceTB/smollm-corpus",
            tokenizer_name="HuggingFaceTB/cosmo2-tokenizer", 
            batch_size=1
        )
        return DataLoader(train_dataset, batch_size=train_dataset.batch_size, shuffle=True,num_workers=9, persistent_workers=True )  # Set num_workers to 9 to use multiple workers for data loading

# Define the LlamaDecoderLayer (based on your earlier example)
class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super(LlamaDecoderLayer, self).__init__()

        self.self_attn = LlamaAttention(hidden_size, num_attention_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(LlamaAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_head_size ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(output)

if __name__ == "__main__":
    # Initialize the model with reduced hidden size and number of layers
    model = LlamaForCausalLM(
        vocab_size=49152,               # Keep the same vocab size
        hidden_size=512,                # Reduced hidden size from 576 to 512 since 512%8 = 0 ->528 -> 
        num_attention_heads=8,          # You may want to reduce this as well if the hidden size is reduced
        num_layers=31,                  # change 30 to 31
        intermediate_size=1648,         # Keep this size (or reduce as well if you need further reduction)
        max_position_embeddings=2048   # No change needed
    ).to(device)

    summary(model)
    # Define checkpoint callback for model saving
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="checkpoint-{step}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_weights_only=True,
        every_n_train_steps=100  # This ensures checkpoints are saved regularly
    )
    # This will log metrics to a CSV file
    csv_logger = CSVLogger("logs", name="my_model")
    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_steps=5000,  # Training for 5000 steps
        accelerator="auto",  # Using CPU (change to "gpu" if available)
        devices=1,  # Use 1 device (CPU or GPU)
        precision=16,  # Use mixed precision if available (can speed up training)
        accumulate_grad_batches=4,  # Gradient accumulation
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        log_every_n_steps=1,  # Log every step
    )
    # Enable progress bar explicitly
    trainer.enable_progress_bar = True
    # Train the model
    trainer.fit(model)

    # Save final checkpoint
    trainer.save_checkpoint("final_model.ckpt")
