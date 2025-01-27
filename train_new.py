import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import logging
#from model_new import LlamaForCausalLM, config_model
from model_smol2 import LlamaForCausalLM, config_model#
from torchinfo import summary
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

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
        "epochs": 1,
        "checkpoints_path": "checkpoints",
        'optimizer': {
            'accumulate_grad_in_fp32': True,
            'clip_grad': 0.05,
            'learning_rate_scheduler': {
                'learning_rate': 1e-4,  # Reduced initial learning rate
                'lr_decay_starting_step': 2000,
                'lr_decay_steps': 1000,  # Decay faster, after 1000 steps instead of 2000
                'lr_decay_style': 'exponential',  # Change to exponential decay (if it suits your task better)
                'lr_warmup_steps': 500,  # Keep warmup steps
                'lr_warmup_style': 'linear',
                'min_decay_lr': 1e-5  # Set min decay lr to a smaller value
            },
            'optimizer_factory': {
                'adam_beta1': 0.9,
                'adam_beta2': 0.9,
                'adam_eps': 1.0e-6,
                'name': 'adamW',
                'torch_adam_is_fused': True
            },
            'weight_decay': 0.001,
            'zero_stage': 0
        }
    }

# Generate text
def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    temperature=0.7,
    top_k=50,
    repetition_penalty=1.2,
    n_gram_block=2,
    device="cpu"
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_tokens = input_ids[0].tolist()  # Track generated tokens for penalties

    with torch.no_grad():
        for step in range(max_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]  # Focus on the last token's logits

            # Apply repetition penalty
            for token_id in set(generated_tokens):
                logits[:, token_id] /= repetition_penalty

            # Apply n-gram blocking
            if len(generated_tokens) >= n_gram_block:
                n_gram = tuple(generated_tokens[-n_gram_block:])
                for token_id in set(generated_tokens):
                    if generated_tokens[-n_gram_block:] == list(n_gram):
                        logits[:, token_id] -= 1e9  # Heavily penalize repeating n-grams

            # Scale logits using temperature
            logits /= temperature

            # Apply Top-K filtering
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)

            # Sample from the filtered probabilities
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            # Append the token to the generated sequence
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop generation if EOS token is produced
            if next_token.item() == tokenizer.eos_token_id:
                print(f"Step {step + 1}: EOS token generated. Stopping generation.")
                break

    # Decode generated tokens into text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
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

# Optimizer and scheduler
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
            # Exponential decay
            decay_rate = 0.96  # Decay factor per step
            decay_steps = current_step - lr_config['lr_decay_starting_step']
            return decay_rate ** decay_steps


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
    torch.set_float32_matmul_precision('high')
    tokenizer, train_loader = prepare_tokenizer_and_dataset(required_config)
    
    # Initialize model
    '''
    rank = 242
    # Apply the weight initialization
    model = LlamaForCausalLM(config_model, rank)
    model.apply(init_weights)
    '''
    model = LlamaForCausalLM(config_model)
    
    print(model)
    summary(model)
    model.to(device)

    optimizer, lr_scheduler = get_optimizer_and_scheduler(model.parameters(), required_config)

    start_epoch, global_step = load_checkpoint(
        f"{required_config['checkpoints_path']}/final_checkpoint.pt", model, optimizer, lr_scheduler
    )

    max_steps = 5000  # Set maximum steps
    if global_step > 0:
        max_steps = 5052
    
    accumulate_steps = 1  # Gradient accumulation

    try:
        for epoch in range(start_epoch, required_config["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{required_config['epochs']}")
            for step, batch in enumerate(train_loader, start=global_step):
                if step == max_steps:
                    logger.info("Reached maximum steps. Exiting training loop.")
                    save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step,
                        loss.item(),
                        f"{required_config['checkpoints_path']}/final_checkpoint.pt",
                    )
                    return

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(input_ids)

                logits = outputs.view(-1, outputs.size(-1))  # Ensure logits are properly shaped
                '''
                logger.debug(f"Step {step}, Logits: {logits[:5]}")  # Log first few logits for inspection

                print(batch["input_ids"][:5])
                print(batch["attention_mask"][:5])
                print(batch["labels"][:5])
                
                print(f"Logits: {logits[:5]}")
                print(f"Logits shape: {logits.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Logits raw values: {logits[:5]}")
                print(f"Logits max: {logits.max()}, Logits min: {logits.min()}")
                print(f"Logits before loss: {logits[:5]}")
                print(f"Labels before loss: {labels[:5]}")
                print(f"Label distribution: {torch.bincount(labels.view(-1))}")
                print(f"NaN in logits: {torch.isnan(logits).any()}")
                print(f"NaN in labels: {torch.isnan(labels).any()}")
                '''

                #loss = torch.nn.functional.mse_loss(logits, labels.view(-1).float())
                loss = torch.nn.functional.cross_entropy(
                    logits, labels.view(-1), ignore_index=tokenizer.pad_token_id
                )
                #print(f"NaN in loss: {torch.isnan(loss).any()}")
                # Check gradients and update optimizer

                if step % accumulate_steps == 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), required_config["optimizer"]["clip_grad"])
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()
                if step % 10 == 0:
                    logger.info(f"Step {step}, Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                sample_prompt = "Gravity is "
                if step % 500 == 0:  # Test text generation every 500 steps
                    logger.info("\n=== Generating Sample Texts ===")
                    for temp in [0.7, 1.0, 1.2, 1.5]:
                        generated = generate_text(
                            model,
                            tokenizer,
                            sample_prompt,
                            max_length=50,
                            temperature=temp,
                            top_k=100,
                            device=device,
                        )
                        logger.info(f"\nPrompt: {sample_prompt}")
                        logger.info(f"Temperature: {temp}")
                        logger.info(f"Generated: {generated}")
                    logger.info("\n=== End of Samples ===\n")
                    model.train()

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted! Saving checkpoint...")
        save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            epoch,
            step,
            loss.item(),
            f"{required_config['checkpoints_path']}/interrupted_checkpoint.pt",
        )

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
