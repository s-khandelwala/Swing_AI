"""
Fine-tune GPT-2 for Golf Swing Feedback Generation

This script fine-tunes a GPT-2 model on golf feedback instruction data.
Optimized for 2x T4 GPUs with DataParallel.

STANDALONE - Not used in main code yet.
Based on LLMs-from-scratch Chapter 7.

Usage:
    python experimental/finetune_golf_feedback_llm.py
    python experimental/finetune_golf_feedback_llm.py --model-size 124M --epochs 3
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add LLMs-from-scratch to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / 'LLMs-from-scratch-main' / 'ch07' / '01_main-chapter-code'))

try:
    from gpt_download import download_and_load_gpt2
    from previous_chapters import (
        GPTModel, load_weights_into_gpt, text_to_token_ids, token_ids_to_text,
        train_model_simple, calc_loss_loader, generate
    )
except ImportError:
    print("⚠️  Could not import from LLMs-from-scratch. Make sure the folder is in the project.")
    print("   Falling back to embedded minimal implementation...")
    # Minimal embedded implementation would go here if needed
    raise


#####################################
# Dataset and Collate Functions
#####################################

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """Collate function for variable-length sequences"""
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


#####################################
# Training Functions (with DataParallel support)
#####################################

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a batch"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100)
    return loss


def train_model_with_dataparallel(
    model, train_loader, val_loader, optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer, use_mixed_precision=True
):
    """Training loop with DataParallel and mixed precision support"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on train and validation sets"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print a sample response"""
    model.eval()
    
    # Handle DataParallel models
    if isinstance(model, nn.DataParallel):
        context_size = model.module.pos_emb.weight.shape[0]
    else:
        context_size = model.pos_emb.weight.shape[0]
    
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=100, context_size=context_size,
            temperature=0.7, top_k=50
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(f"\n📝 Sample Generation:\n{decoded_text}\n")
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path="experimental/golf_feedback_loss_plot.pdf"):
    """Plot training and validation losses"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    
    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"📊 Loss plot saved as {save_path}")


#####################################
# Main Training Function
#####################################

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 for Golf Feedback')
    parser.add_argument('--model-size', type=str, default='124M', 
                       choices=['124M', '355M', '774M', '1558M'],
                       help='GPT-2 model size (default: 124M)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size per GPU (default: 8)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--data-dir', type=str, default='experimental/golf_feedback_finetuning',
                       help='Directory containing training data JSON files')
    parser.add_argument('--output-dir', type=str, default='experimental/golf_feedback_models',
                       help='Directory to save fine-tuned model')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--eval-freq', type=int, default=10,
                       help='Evaluation frequency in steps (default: 10)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Golf Feedback LLM Fine-tuning")
    print("=" * 70)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    print(f"\n🖥️  Device: {device}")
    print(f"🚀 GPUs available: {num_gpus}")
    if num_gpus > 1:
        print(f"   Using DataParallel with {num_gpus} GPUs")
        effective_batch_size = args.batch_size * num_gpus
        print(f"   Effective batch size: {effective_batch_size} (per-GPU: {args.batch_size})")
    else:
        print(f"   Using single GPU")
        effective_batch_size = args.batch_size
    
    # Load training data
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'golf_feedback_train.json'
    val_path = data_dir / 'golf_feedback_val.json'
    
    if not train_path.exists():
        print(f"\n❌ Training data not found at {train_path}")
        print(f"   Please run: python experimental/generate_golf_feedback_training_data.py")
        return
    
    print(f"\n📂 Loading training data...")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"   Training examples: {len(train_data)}")
    print(f"   Validation examples: {len(val_data)}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    
    # Create data loaders
    customized_collate_fn = partial(
        custom_collate_fn, 
        device=device, 
        allowed_max_length=1024
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=customized_collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=customized_collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model configuration
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    
    model_configs = {
        "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    BASE_CONFIG.update(model_configs[args.model_size])
    
    print(f"\n🤖 Loading GPT-2 {args.model_size}...")
    models_dir = "experimental/gpt2_models"
    os.makedirs(models_dir, exist_ok=True)
    
    settings, params = download_and_load_gpt2(model_size=args.model_size, models_dir=models_dir)
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)
    
    # Wrap with DataParallel if multiple GPUs
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"✅ Model wrapped with DataParallel")
    
    print(f"✅ Model loaded: GPT-2 {args.model_size}")
    
    # Initial losses
    print(f"\n📊 Initial losses:")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"   Training loss: {train_loss:.4f}")
    print(f"   Validation loss: {val_loss:.4f}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    
    # Training
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Mixed precision: {not args.no_mixed_precision}")
    
    start_time = time.time()
    
    # Use custom training function with DataParallel support
    train_losses, val_losses, tokens_seen = train_model_with_dataparallel(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, 
        eval_freq=args.eval_freq, 
        eval_iter=5,
        start_context=format_input(val_data[0]), 
        tokenizer=tokenizer,
        use_mixed_precision=not args.no_mixed_precision
    )
    
    end_time = time.time()
    training_time_minutes = (end_time - start_time) / 60
    
    print(f"\n✅ Training completed in {training_time_minutes:.2f} minutes")
    
    # Plot losses
    epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"gpt2-{args.model_size}-golf-feedback.pth"
    model_path = output_dir / model_name
    
    # Save model (unwrap DataParallel if needed)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    print(f"\n💾 Model saved to {model_path}")
    print(f"\n🎉 Fine-tuning complete!")
    print(f"   Final train loss: {train_losses[-1]:.4f}")
    print(f"   Final val loss: {val_losses[-1]:.4f}")


if __name__ == '__main__':
    main()
