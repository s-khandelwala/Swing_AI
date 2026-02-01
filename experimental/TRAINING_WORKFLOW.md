# Golf Feedback LLM Training Workflow

## Overview

This document explains how the golf feedback LLM fine-tuning system works, from data generation to model deployment.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Knowledge Base (knowledge/golf_instruction/)            │
│    - setup/                                                 │
│    - tempo_rhythm/                                          │
│    - weight_shift/                                           │
│    - body_rotation/                                         │
│    - followthrough/                                          │
│    - balance/                                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Data Generation                                          │
│    generate_golf_feedback_training_data.py                  │
│    - Reads instruction files                                │
│    - Creates instruction-response pairs                     │
│    - Generates synthetic training examples                 │
│    - Outputs: train/val/test JSON files                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Model Fine-tuning                                        │
│    complete_golf_feedback_finetuning.py                     │
│    - Downloads GPT-2 weights (124M/355M/774M/1558M)         │
│    - Loads pre-trained GPT-2                                │
│    - Fine-tunes on golf feedback data                      │
│    - Saves fine-tuned model                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Fine-tuned Model                                         │
│    gpt2-{size}-golf-feedback.pth                            │
│    - Specialized for golf swing feedback                    │
│    - Ready for inference                                    │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Training Process

### Step 1: Generate Training Data

**Script:** `generate_golf_feedback_training_data.py`

**What it does:**
1. Scans `knowledge/golf_instruction/` for instruction files
2. For each golf area (setup, tempo, weight_shift, etc.):
   - Loads instruction content
   - Generates synthetic examples with varying scores
   - Creates instruction-response pairs
3. Splits data: 85% train, 10% val, 5% test
4. Saves JSON files to `experimental/golf_feedback_finetuning/`

**Example generated entry:**
```json
{
  "instruction": "Provide personalized golf swing feedback...",
  "input": "Area: setup_quality\nCurrent Score: 0.45\nTarget Score: 0.75\n...",
  "output": "WHAT SHOULD HAPPEN (Ideal/Expected):\nSetup Quality...\n\nWHAT IS HAPPENING (Current Issue):\nYour setup quality score is 45.0%...\n\nWHAT TO DO (Actionable Steps):\nBased on professional golf instruction..."
}
```

**Output files:**
- `golf_feedback_train.json` (~340 examples)
- `golf_feedback_val.json` (~40 examples)
- `golf_feedback_test.json` (~20 examples)

### Step 2: Fine-tune GPT-2

**Script:** `complete_golf_feedback_finetuning.py`

**What it does:**

#### 2.1 Model Setup
1. **Downloads GPT-2 weights** (if not already downloaded):
   - Downloads from OpenAI's official repository
   - Falls back to backup URL if needed
   - Stores in `experimental/gpt2_models/{model_size}/`

2. **Loads GPT-2 architecture:**
   - Creates `GPTModel` with specified size (124M/355M/774M/1558M)
   - Loads pre-trained weights from TensorFlow checkpoint
   - Converts to PyTorch format

3. **Multi-GPU Setup:**
   - Detects available GPUs
   - Wraps model with `nn.DataParallel` if multiple GPUs
   - Effective batch size = `batch_size × num_gpus`

#### 2.2 Data Loading
1. **Loads JSON training data**
2. **Tokenizes with GPT-2 tokenizer** (tiktoken)
3. **Creates DataLoaders:**
   - Custom collate function handles variable-length sequences
   - Pads sequences to batch max length
   - Sets padding tokens to ignore_index in loss

#### 2.3 Training Loop

**For each epoch:**
```
For each batch:
  1. Forward pass (with mixed precision if enabled)
     - Model predicts next token probabilities
     - Cross-entropy loss computed (ignoring padding)
  
  2. Backward pass
     - Compute gradients
     - Update weights with AdamW optimizer
  
  3. Evaluation (every N steps)
     - Compute train/val loss
     - Log progress
  
  4. Sample generation (end of epoch)
     - Generate example feedback
     - Print to console
```

**Key features:**
- **Mixed Precision (FP16):** Faster training, lower memory
- **Gradient Accumulation:** Not used by default (can be added)
- **Learning Rate:** 5e-5 (AdamW with weight decay 0.1)
- **Context Length:** 1024 tokens max

#### 2.4 Model Saving
1. Unwraps DataParallel if used
2. Saves PyTorch state dict
3. Saves loss plot

**Output:**
- `gpt2-{size}-golf-feedback.pth` - Fine-tuned model
- `loss_plot_{size}.pdf` - Training curves

## Training Time Breakdown (2x T4 GPUs)

### GPT-2 Small (124M) - Recommended

| Phase | Time | Details |
|-------|------|---------|
| **Data Generation** | < 1 min | Reads knowledge files, generates examples |
| **Model Download** | 2-5 min | First run only (500MB download) |
| **Weight Loading** | < 10 sec | Converts TF → PyTorch |
| **Training (400 ex, 2 epochs)** | 2-5 min | With DataParallel, FP16 |
| **Training (1000 ex, 2 epochs)** | 5-10 min | More examples |
| **Total (first run)** | 5-10 min | Including download |
| **Total (subsequent)** | 2-10 min | No download needed |

### GPT-2 Medium (355M)

| Phase | Time | Details |
|-------|------|---------|
| **Model Download** | 5-10 min | First run only (1.4GB download) |
| **Training (400 ex, 2 epochs)** | 5-10 min | With DataParallel, FP16 |
| **Training (1000 ex, 2 epochs)** | 10-20 min | More examples |
| **Total (first run)** | 10-20 min | Including download |
| **Total (subsequent)** | 5-20 min | No download needed |

## Memory Usage (2x T4 GPUs, 16GB each)

| Model Size | Per-GPU Memory | Batch Size | Status |
|------------|----------------|------------|--------|
| 124M       | ~2-3 GB        | 8          | ✅ Comfortable |
| 355M       | ~4-6 GB        | 8          | ✅ Comfortable |
| 774M       | ~8-10 GB       | 4-6        | ⚠️  Tight |
| 1558M      | ~12-14 GB      | 2-4        | ❌ May not fit |

## Training Configuration

### Default Settings
```python
model_size = "124M"          # GPT-2 small
epochs = 2                   # Training epochs
batch_size = 8               # Per GPU
learning_rate = 5e-5         # AdamW learning rate
weight_decay = 0.1           # L2 regularization
mixed_precision = True       # FP16 training
eval_freq = 10               # Evaluate every 10 steps
context_length = 1024        # Max sequence length
```

### Custom Configuration
```bash
python experimental/complete_golf_feedback_finetuning.py \
    --model-size 124M \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-5 \
    --eval-freq 10
```

## How Training Works (Technical Details)

### 1. Instruction Formatting
```
Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

### Instruction:
Provide personalized golf swing feedback...

### Input:
Area: setup_quality
Current Score: 0.45
Target Score: 0.75
...

### Response:
WHAT SHOULD HAPPEN (Ideal/Expected):
...
```

### 2. Tokenization
- Uses GPT-2 tokenizer (BPE encoding)
- Vocabulary size: 50,257 tokens
- Special tokens: `<|endoftext|>` (50256)

### 3. Model Forward Pass
```
Input tokens → Token Embeddings → Position Embeddings
→ Transformer Blocks (12-48 layers)
→ Layer Norm → Output Head → Logits (vocab_size)
```

### 4. Loss Calculation
- **Cross-entropy loss** on next-token prediction
- **Ignore padding tokens** (ignore_index=-100)
- **Average over all non-padding positions**

### 5. Optimization
- **Optimizer:** AdamW (β₁=0.9, β₂=0.999)
- **Learning Rate:** 5e-5 (fixed, no scheduler by default)
- **Weight Decay:** 0.1 (L2 regularization)
- **Gradient Clipping:** Not used (can be added)

### 6. Mixed Precision (FP16)
- Forward pass in FP16 (faster, less memory)
- Loss scaling to prevent underflow
- Backward pass in FP32 (numerical stability)
- ~2x speedup, ~50% memory reduction

### 7. DataParallel
- Splits batch across GPUs
- Each GPU processes `batch_size / num_gpus` samples
- Gradients averaged across GPUs
- Effective batch size = `batch_size × num_gpus`

## Evaluation Metrics

### Loss
- **Training Loss:** Cross-entropy on training set
- **Validation Loss:** Cross-entropy on validation set
- **Target:** Should decrease over epochs (typically 3.0 → 1.5 → 0.8)

### Sample Generation
- Generates feedback after each epoch
- Uses temperature=0.7, top_k=50 for diversity
- Checks if output is coherent and golf-specific

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (e.g., `--batch-size 4`)
- Use smaller model (`--model-size 124M`)
- Disable mixed precision: `--no-mixed-precision`

### Slow Training
- Ensure DataParallel is working (check `nvidia-smi`)
- Use mixed precision (default enabled)
- Reduce number of examples
- Use smaller model

### Poor Quality Output
- Train for more epochs (`--epochs 3-5`)
- Generate more training data
- Use larger model (`--model-size 355M`)
- Check if training loss is decreasing

## Next Steps After Training

1. **Test the model:**
   ```python
   # Load fine-tuned model
   model = GPTModel(BASE_CONFIG)
   model.load_state_dict(torch.load('experimental/golf_feedback_models/gpt2-124M-golf-feedback.pth'))
   model.eval()
   
   # Generate feedback
   input_text = format_input(entry)
   encoded = text_to_token_ids(input_text, tokenizer)
   token_ids = generate(model, encoded, max_new_tokens=200, context_size=1024, temperature=0.7)
   response = token_ids_to_text(token_ids, tokenizer)
   ```

2. **Integrate into main system:**
   - Create `FineTunedGolfFeedbackGenerator` class
   - Extend `LLMFeedbackGenerator`
   - Replace Ollama calls with fine-tuned model

3. **Iterate:**
   - Add more training examples
   - Fine-tune hyperparameters
   - Collect real user feedback for training
