# Golf Feedback LLM Fine-tuning (Experimental)

**⚠️ STANDALONE - Not integrated into main codebase yet**

This directory contains experimental scripts for fine-tuning a GPT-2 model specifically for golf swing feedback generation.

## Overview

Instead of using Ollama with a general-purpose LLM, these scripts allow you to:
1. Generate training data from your golf instruction knowledge base
2. Fine-tune GPT-2 (small/medium) on golf-specific feedback
3. Get a specialized model optimized for golf swing analysis

## Files

- **`generate_golf_feedback_training_data.py`**: Generates instruction-response pairs from knowledge base
- **`complete_golf_feedback_finetuning.py`**: **COMPLETE** self-contained fine-tuning script (all dependencies included)
- **`finetune_golf_feedback_llm.py`**: Alternative script (requires LLMs-from-scratch folder)
- **`TRAINING_WORKFLOW.md`**: Detailed explanation of how training works
- **`README.md`**: This file

## Quick Start (Recommended)

### Step 1: Generate Training Data

```bash
python experimental/generate_golf_feedback_training_data.py
```

### Step 2: Fine-tune Model (Complete Script)

```bash
# Uses complete self-contained script (recommended)
python experimental/complete_golf_feedback_finetuning.py --model-size 124M --epochs 2
```

This script includes **all necessary functions** embedded (like `complete_event_detector_training.py`).

## Prerequisites

1. **LLMs-from-scratch folder**: The `LLMs-from-scratch-main` folder must be in the project root
2. **Dependencies**: 
   ```bash
   pip install torch tiktoken matplotlib numpy tqdm requests
   ```
3. **Knowledge Base**: Your `knowledge/golf_instruction/` folder should contain instruction files

## Quick Start

### Step 1: Generate Training Data

```bash
python experimental/generate_golf_feedback_training_data.py
```

This will:
- Read golf instruction files from `knowledge/golf_instruction/`
- Generate ~400 training examples (50 per area × 8 areas)
- Create train/val/test splits (85/10/5)
- Save JSON files to `experimental/golf_feedback_finetuning/`

**Output:**
- `golf_feedback_train.json` - Training set
- `golf_feedback_val.json` - Validation set
- `golf_feedback_test.json` - Test set

### Step 2: Fine-tune Model

```bash
# Default: GPT-2 small (124M), 2 epochs, batch_size=8 per GPU
python experimental/finetune_golf_feedback_llm.py

# Custom configuration
python experimental/finetune_golf_feedback_llm.py \
    --model-size 124M \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-5
```

**Options:**
- `--model-size`: `124M` (recommended), `355M`, `774M`, `1558M`
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Batch size per GPU (default: 8)
- `--lr`: Learning rate (default: 5e-5)
- `--no-mixed-precision`: Disable FP16 training

**With 2x T4 GPUs:**
- Automatically uses DataParallel
- Effective batch size = `batch_size × num_gpus`
- Example: `--batch-size 8` → 16 total with 2 GPUs

**Output:**
- Fine-tuned model: `experimental/golf_feedback_models/gpt2-{size}-golf-feedback.pth`
- Loss plot: `experimental/golf_feedback_loss_plot.pdf`

## Training Time Estimates (2x T4 GPUs)

| Model Size | Examples | Epochs | Estimated Time |
|------------|----------|--------|----------------|
| 124M       | 400      | 2      | 2-5 minutes    |
| 124M       | 1000     | 2      | 5-10 minutes   |
| 355M       | 400      | 2      | 5-10 minutes   |
| 355M       | 1000     | 2      | 10-20 minutes  |

## Usage After Training

Once you have a fine-tuned model, you can load it for inference:

```python
import torch
from experimental.finetune_golf_feedback_llm import GPTModel, load_weights_into_gpt, format_input, generate, text_to_token_ids, token_ids_to_text
import tiktoken

# Load model
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load('experimental/golf_feedback_models/gpt2-124M-golf-feedback.pth'))
model.eval()

# Generate feedback
entry = {
    "instruction": "Provide personalized golf swing feedback...",
    "input": "Area: setup_quality\nCurrent Score: 0.45\n..."
}
input_text = format_input(entry)
encoded = text_to_token_ids(input_text, tokenizer)
token_ids = generate(model, encoded, max_new_tokens=200, context_size=1024, temperature=0.7)
response = token_ids_to_text(token_ids, tokenizer)
print(response)
```

## Integration with Main Codebase

**Currently NOT integrated.** To integrate:

1. Create a new class `FineTunedGolfFeedbackGenerator` that extends `LLMFeedbackGenerator`
2. Load the fine-tuned model instead of using Ollama
3. Update `swing_improvement_system.py` to use the fine-tuned model

## Notes

- **Model Size**: Start with `124M` (GPT-2 small) for fast iteration
- **Data Quality**: The generated training data is synthetic. For better results, consider:
  - Adding real feedback examples
  - Using your existing RAG system to generate more diverse examples
  - Fine-tuning on actual user feedback if available
- **Memory**: GPT-2 124M fits easily in T4 (16GB). 355M also works. 774M+ may be tight.
- **Mixed Precision**: Enabled by default for faster training and lower memory usage

## Troubleshooting

**Import Error: "Could not import from LLMs-from-scratch"**
- Make sure `LLMs-from-scratch-main` folder is in project root
- Check that `ch07/01_main-chapter-code/previous_chapters.py` exists

**CUDA Out of Memory**
- Reduce `--batch-size` (e.g., `--batch-size 4`)
- Use smaller model (`--model-size 124M`)
- Disable mixed precision: `--no-mixed-precision`

**Training too slow**
- Use smaller model (`124M` instead of `355M`)
- Reduce number of examples
- Ensure DataParallel is working (check GPU usage with `nvidia-smi`)
