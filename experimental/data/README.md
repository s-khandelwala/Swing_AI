# Training Data Folder

This folder contains the training data for fine-tuning the golf feedback LLM.

## Files

- **`golf_feedback_train.json`** - Training set (used for model training)
- **`golf_feedback_val.json`** - Validation set (used for evaluation during training)
- **`golf_feedback_test.json`** - Test set (optional, for final evaluation)

## Data Format

Each JSON file contains an array of training examples. Each example has the format:

```json
{
  "instruction": "Provide personalized golf swing feedback based on quality scores and golf instruction principles.",
  "input": "Area: setup_quality\nCurrent Score: 0.45\nTarget Score: 0.75\nAll Quality Scores: ['0.45', '0.72', '0.68', ...]\nSwing Type: balanced_swing",
  "output": "WHAT SHOULD HAPPEN (Ideal/Expected):\n...\n\nWHAT IS HAPPENING (Current Issue):\n...\n\nWHAT TO DO (Actionable Steps):\n..."
}
```

## Current Data

- **Training examples:** 6 (sample data)
- **Validation examples:** 2 (sample data)

## Adding More Data

To add more training examples:

1. **Manual:** Edit the JSON files directly
2. **Generate:** Run `python experimental/generate_golf_feedback_training_data.py` (if knowledge base is available)
3. **Automatic:** The training script will use whatever data is in this folder

## Notes

- The script automatically looks for data in `experimental/data/` first
- If data is not found, it will check `experimental/golf_feedback_finetuning/`
- More training examples = better model performance (typically 100-1000+ examples recommended)
