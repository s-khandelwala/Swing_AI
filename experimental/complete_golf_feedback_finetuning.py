"""
Complete Golf Feedback LLM Fine-tuning Script
All dependencies included in this single file.
Optimized for 2x T4 GPUs with DataParallel.

STANDALONE - Not used in main code yet.

Usage:
    python experimental/complete_golf_feedback_finetuning.py
    python experimental/complete_golf_feedback_finetuning.py --model-size 124M --epochs 3 --batch-size 8
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import json
import os
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

# Try to import tensorflow for GPT-2 weight loading (optional, will use alternative if not available)
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not available. Will attempt to download pre-converted weights if needed.")


# ============================================================================
# GPT-2 MODEL ARCHITECTURE
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ============================================================================
# GPT-2 WEIGHT LOADING
# ============================================================================

def download_file(url, destination, backup_url=None):
    """Download a file with progress bar and backup URL support"""
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"✅ File already exists: {os.path.basename(destination)}")
                return True

        block_size = 1024
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"⚠️  Primary URL failed. Trying backup: {os.path.basename(backup_url)}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"❌ Failed to download from both URLs.\n"
            f"   Primary: {url}\n"
            f"   Backup: {backup_url}\n"
            "Check your internet connection."
        )
        raise RuntimeError(error_message)


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Load GPT-2 parameters from TensorFlow checkpoint"""
    if not HAS_TF:
        raise RuntimeError("TensorFlow required for loading GPT-2 weights. Install with: pip install tensorflow")
    
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def download_and_load_gpt2(model_size, models_dir):
    """Download and load GPT-2 model weights"""
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    print(f"📥 Downloading GPT-2 {model_size} weights...")
    os.makedirs(model_dir, exist_ok=True)
    
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        backup_url = f"{backup_base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    if not HAS_TF:
        raise RuntimeError(
            "TensorFlow required for loading GPT-2 weights.\n"
            "Install with: pip install tensorflow\n"
            "Or use a pre-converted PyTorch checkpoint if available."
        )

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def assign(left, right):
    """Assign numpy array to PyTorch parameter"""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """Load GPT-2 weights into model"""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# ============================================================================
# DATASET AND DATA LOADING
# ============================================================================

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
    """Format instruction and input for the model"""
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
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs"""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs to text"""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a batch"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over a data loader"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generate text from model"""
    # Handle DataParallel models
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = actual_model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def train_model_with_dataparallel(
    model, train_loader, val_loader, optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer, use_mixed_precision=True
):
    """Training loop with DataParallel and mixed precision support"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    for epoch in range(num_epochs):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for input_batch, target_batch in pbar:
            optimizer.zero_grad()
            
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

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.3f}',
                    'val_loss': f'{val_loss:.3f}'
                })
                print(f"\nEp {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Generate sample after each epoch
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
    
    if isinstance(model, nn.DataParallel):
        context_size = model.module.pos_emb.weight.shape[0]
    else:
        context_size = model.pos_emb.weight.shape[0]
    
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=150, context_size=context_size,
            temperature=0.7, top_k=50
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(f"\n📝 Sample Generation:\n{decoded_text}\n")
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path):
    """Plot training and validation losses"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(epochs_seen, train_losses, label="Training loss", linewidth=2)
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", linewidth=2)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen", fontsize=12)
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Loss plot saved as {save_path}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    # Path configuration (matching Kaggle file structure)
    DATASET_NAME = "swingai-model"
    INPUT_DIR = f"/kaggle/input/{DATASET_NAME}" if os.path.exists("/kaggle") else "experimental"
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "experimental"
    MODELS_DIR = f"{OUTPUT_DIR}/models"
    DATA_DIR = f"{INPUT_DIR}/data" if os.path.exists("/kaggle") else f"{INPUT_DIR}/data"
    GPT2_MODELS_DIR = f"{OUTPUT_DIR}/gpt2_models"
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GPT2_MODELS_DIR, exist_ok=True)
    
    # Default configuration (can be overridden with command-line args if needed)
    # For auto-run, we use defaults
    import sys
    
    # Check if running from command line with args, otherwise use defaults
    if len(sys.argv) > 1:
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
        parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                           help='Directory containing training data JSON files')
        parser.add_argument('--output-dir', type=str, default=MODELS_DIR,
                           help='Directory to save fine-tuned model')
        parser.add_argument('--models-dir', type=str, default=GPT2_MODELS_DIR,
                           help='Directory to store/download GPT-2 weights')
        parser.add_argument('--no-mixed-precision', action='store_true',
                           help='Disable mixed precision training')
        parser.add_argument('--eval-freq', type=int, default=10,
                           help='Evaluation frequency in steps (default: 10)')
        args = parser.parse_args()
    else:
        # Defaults for auto-run
        class Args:
            model_size = '124M'
            epochs = 2
            batch_size = 8
            lr = 5e-5
            data_dir = DATA_DIR
            output_dir = MODELS_DIR
            models_dir = GPT2_MODELS_DIR
            no_mixed_precision = False
            eval_freq = 10
        args = Args()
    
    print("=" * 70)
    print("Golf Feedback LLM Fine-tuning (Complete Self-Contained Script)")
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
        print(f"\n⚠️  Training data not found at {train_path}")
        # Try alternative locations (local fallback)
        alt_paths = [
            Path('experimental/data'),
            Path(f'{INPUT_DIR}/data'),
            Path(f'{OUTPUT_DIR}/data'),
        ]
        
        found = False
        for alt_data_dir in alt_paths:
            alt_train_path = alt_data_dir / 'golf_feedback_train.json'
            alt_val_path = alt_data_dir / 'golf_feedback_val.json'
            
            if alt_train_path.exists():
                train_path = alt_train_path
                val_path = alt_val_path
                print(f"   Found data at: {alt_data_dir}")
                found = True
                break
        
        if not found:
            print(f"\n❌ No training data found. Please ensure data files exist in:")
            print(f"   - {data_dir}/golf_feedback_train.json")
            print(f"   - {data_dir}/golf_feedback_val.json")
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
    print(f"   Models directory: {args.models_dir}")
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        settings, params = download_and_load_gpt2(model_size=args.model_size, models_dir=models_dir)
    except Exception as e:
        print(f"❌ Error loading GPT-2 weights: {e}")
        print(f"   Make sure TensorFlow is installed: pip install tensorflow")
        return
    
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
    plot_path = os.path.join(args.output_dir, f"loss_plot_{args.model_size}.pdf")
    epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_path)
    
    # Save model (using same naming convention as Kaggle files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"gpt2-{args.model_size}-golf-feedback.pth.tar"
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
