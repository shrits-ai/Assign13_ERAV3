import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os
import logging
from model_smol2 import LlamaForCausalLM, config_model
from torchinfo import summary

# Set up logging
def setup_logging(log_file="training.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    log_format = logging.Formatter("%(asctime)s - %(message)s")
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# Configuration
def get_config():
    return {
        "batch_size": 8,
        "sequence_length": 256, #Reducing from 2048 -> 256 due to memory issue
        "epochs": 5,
        "clip_grad": 1.0,
        "checkpoints_path": "checkpoints",
        'optimizer': {
            'accumulate_grad_in_fp32': True,
            'clip_grad': 1.0,
            'learning_rate_scheduler': {
                'learning_rate': 0.0001,
                'lr_decay_starting_step': 3000,
                'lr_decay_steps': 1000,
                'lr_decay_style': 'linear',
                'lr_warmup_steps': 2000,
                'lr_warmup_style': 'linear',
                'min_decay_lr': 0
            },
            'optimizer_factory': {
                'adam_beta1': 0.9,
                'adam_beta2': 0.95,
                'adam_eps': 1.0e-08,
                'name': 'adamW',
                'torch_adam_is_fused': True
            },
            'weight_decay': 0.01,
            'zero_stage': 0
        }
    }

# Generate text
def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, device="cpu"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)

            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text

# Checkpoint utilities
def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
            "step": step,
        },
        path,
    )

def load_checkpoint(path, model, optimizer, scheduler):
    if os.path.exists(path):
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["step"]
    return 0, 0

# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Tokenizer and Dataset preparation
def prepare_tokenizer_and_dataset(required_config):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True, split="train"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=required_config["sequence_length"], padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor(
            [item["attention_mask"] for item in batch], dtype=torch.long
        )
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    data_loader = DataLoader(
        tokenized_dataset, batch_size=required_config["batch_size"], collate_fn=collate_fn
    )

    return tokenizer, data_loader

# Apply pruning
def apply_pruning(model, pruning_amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.random_unstructured(module, name="weight", amount=pruning_amount)

def get_optimizer_and_scheduler(model_parameters, config):
    # Extract optimizer config
    optimizer_config = config['optimizer']['optimizer_factory']
    weight_decay = config['optimizer']['weight_decay']

    # Initialize optimizer
    optimizer = AdamW(
        model_parameters,
        lr=config['optimizer']['learning_rate_scheduler']['learning_rate'],
        betas=(optimizer_config['adam_beta1'], optimizer_config['adam_beta2']),
        eps=optimizer_config['adam_eps'],
        weight_decay=weight_decay,
        fused=optimizer_config['torch_adam_is_fused']
    )

    # Extract learning rate scheduler config
    lr_config = config['optimizer']['learning_rate_scheduler']

    # Define a lambda function for learning rate adjustment
    def lr_lambda(current_step):
        if current_step < lr_config['lr_warmup_steps']:
            # Linear warmup
            return current_step / lr_config['lr_warmup_steps']
        elif current_step < lr_config['lr_decay_starting_step']:
            # Constant learning rate before decay
            return 1.0
        else:
            # Linear decay
            decay_steps = lr_config['lr_decay_steps']
            decay_start = lr_config['lr_decay_starting_step']
            min_lr_ratio = lr_config['min_decay_lr'] / lr_config['learning_rate']
            decay_factor = max(
                min_lr_ratio,
                (1 - (current_step - decay_start) / decay_steps)
            )
            return decay_factor

    # Initialize scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

# Main function
def main():
    # Setup
    logger = setup_logging()
    required_config = get_config()
    os.makedirs(required_config["checkpoints_path"], exist_ok=True)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    tokenizer, train_loader = prepare_tokenizer_and_dataset(required_config)

    # Initialize model
    rank = 242
    model = LlamaForCausalLM(config_model, rank)
    #apply_pruning(model, pruning_amount=0.375)
    print(model)
    summary(model)
    model.to(device)

    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model.parameters(), required_config)

    start_epoch, global_step = load_checkpoint(
        f"{required_config['checkpoints_path']}/checkpoint_step_5000.pt", model, optimizer, scheduler
    )

    

    # Training loop
    model.train()
    try:
        for epoch in range(start_epoch, required_config["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{required_config['epochs']}")
            for step, batch in enumerate(train_loader, start=global_step):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                logits = outputs.view(-1, tokenizer.vocab_size)

                loss = torch.nn.functional.cross_entropy(
                    logits, labels.view(-1), label_smoothing=0.1
                )

                loss.backward()
                # Clip gradients if required
                if required_config['optimizer']['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), required_config['optimizer']['clip_grad'])
                optimizer.zero_grad()
                optimizer.step()
                
                scheduler.step()

                if step % 100 == 0:
                    logger.info(
                        f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                if step % 100 == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        loss.item(),
                        f"{required_config['checkpoints_path']}/latest_checkpoint.pt",
                    )

                    if step % 1000 == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            step,
                            loss.item(),
                            f"{required_config['checkpoints_path']}/checkpoint_step_{step}.pt",
                        )

                if step % 500 == 0:
                    logger.info("\n=== Generating Sample Texts ===")
                    sample_prompt = "Gravity is "
                    for temp in [0.7, 1.0]:
                        generated = generate_text(
                            model,
                            tokenizer,
                            sample_prompt,
                            temperature=temp,
                            max_length=10,
                            device=device,
                        )
                        logger.info(f"\nPrompt: {sample_prompt}")
                        logger.info(f"Temperature: {temp}")
                        logger.info(f"Generated: {generated}")
                    model.train()

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                step,
                loss.item(),
                f"{required_config['checkpoints_path']}/checkpoint_epoch_{epoch+1}.pt",
            )

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted! Saving checkpoint...")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            loss.item(),
            f"{required_config['checkpoints_path']}/interrupted_checkpoint.pt",
        )

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
