import os
import torch
from torchinfo import summary
import pytorch_lightning as pl
from transformers import AutoTokenizer
import train

# Reuse the LlamaForCausalLM, LlamaDecoderLayer, LlamaAttention, and StreamingDataset classes from above

if __name__ == "__main__":
    # Define the path to the last checkpoint
    checkpoint_path = "checkpoints/checkpoint-5000.ckpt"

    # Initialize the model configuration
    model = train.LlamaForCausalLM(
        vocab_size=49152,               # Same vocab size
        hidden_size=512,                # Keep hidden size fixed
        num_attention_heads=8,          # Keep this fixed
        num_layers=31,                  # Same as the trained model
        intermediate_size=1648,         # Same intermediate size
        max_position_embeddings=2048   # No change
    )

    # Load the checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = model.load_from_checkpoint(checkpoint_path)  # Use PyTorch Lightning's checkpoint loading
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Ensure you have the correct path.")

    # Print the model summary
    summary(model)

    # Define checkpoint callback for saving new checkpoints
    new_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="new_checkpoints/",
        filename="checkpoint-{step}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_weights_only=True
    )

    # Initialize PyTorch Lightning Trainer for 50 more steps
    trainer = pl.Trainer(
        max_steps=5050,  # Train for 50 more steps
        accelerator="auto",  # Automatically use GPU if available
        devices=1,  # Use one device (GPU or CPU)
        precision=16,  # Use mixed precision
        accumulate_grad_batches=4,  # Gradient accumulation
        callbacks=[new_checkpoint_callback],
        log_every_n_steps=2  # Log every 2 steps for more frequent updates
    )

    # Enable progress bar explicitly
    trainer.enable_progress_bar = True

    # Resume training
    print("Resuming training for 50 more steps...")
    trainer.fit(model)

    # Save final model checkpoint after 5050 steps
    trainer.save_checkpoint("final_model_extended.ckpt")
    print("Training complete. Final checkpoint saved as 'final_model_extended.ckpt'.")
