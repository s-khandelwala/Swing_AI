"""
Generate training data for golf feedback fine-tuning from knowledge base.

This script creates instruction-response pairs from your golf instruction
knowledge base for fine-tuning an LLM on golf swing feedback.

STANDALONE - Not used in main code yet.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict

# Golf instruction areas matching your quality scores
AREA_MAPPING = {
    'setup_quality': 'setup',
    'tempo_rhythm': 'tempo_rhythm',
    'weight_shift': 'weight_shift',
    'body_rotation': 'body_rotation',
    'impact_quality': 'impact',  # May need to create this
    'followthrough': 'followthrough',
    'balance': 'balance',
    'consistency': 'tempo_rhythm'  # Can use tempo for consistency
}

AREA_DESCRIPTIONS = {
    'setup_quality': 'Setup Quality (stance, posture, alignment)',
    'tempo_rhythm': 'Tempo & Rhythm (smooth, unhurried transition)',
    'weight_shift': 'Weight Shift (moves to front foot in downswing)',
    'body_rotation': 'Body Rotation (full shoulder turn, proper sequence)',
    'impact_quality': 'Impact Quality (inferred from swing mechanics)',
    'followthrough': 'Follow-Through (full extension, weight on lead foot)',
    'balance': 'Balance (hold finish position, stable throughout)',
    'consistency': 'Consistency (repeatable motion and results)'
}


def load_knowledge_file(area: str) -> str:
    """Load knowledge content for a given area"""
    knowledge_path = Path('knowledge/golf_instruction')
    
    # Map area to directory
    area_dir_map = {
        'setup_quality': 'setup',
        'tempo_rhythm': 'tempo_rhythm',
        'weight_shift': 'weight_shift',
        'body_rotation': 'body_rotation',
        'followthrough': 'followthrough',
        'balance': 'balance',
    }
    
    dir_name = area_dir_map.get(area, area)
    area_dir = knowledge_path / dir_name
    
    # Try to find any .txt file in the directory
    txt_files = list(area_dir.glob('*.txt'))
    if txt_files:
        return txt_files[0].read_text(encoding='utf-8')
    
    return ""


def create_feedback_response(area: str, score: float, knowledge: str) -> str:
    """Create a feedback response based on area, score, and knowledge"""
    
    # Determine feedback tone based on score
    if score < 0.4:
        severity = "major"
        issue_desc = "significant issues"
        urgency = "needs immediate attention"
    elif score < 0.6:
        severity = "moderate"
        issue_desc = "needs improvement"
        urgency = "should be addressed"
    else:
        severity = "minor"
        issue_desc = "could be refined"
        urgency = "fine-tuning opportunity"
    
    # Extract key points from knowledge (first 300 chars)
    knowledge_summary = knowledge[:300].replace('\n', ' ').strip()
    
    # Create structured feedback
    response = f"""WHAT SHOULD HAPPEN (Ideal/Expected):
{AREA_DESCRIPTIONS[area]}. {knowledge_summary}

WHAT IS HAPPENING (Current Issue):
Your {area.replace('_', ' ')} score is {score:.1%}, indicating {issue_desc}. This {urgency} to improve your swing effectiveness.

WHAT TO DO (Actionable Steps):
Based on professional golf instruction principles, focus on the fundamentals from the knowledge base. Practice with specific drills targeting {area.replace('_', ' ')} improvement."""
    
    return response


def generate_training_examples(num_examples_per_area: int = 50) -> List[Dict]:
    """Generate training examples from knowledge base"""
    training_data = []
    
    areas = list(AREA_DESCRIPTIONS.keys())
    
    for area in areas:
        # Load knowledge for this area
        knowledge = load_knowledge_file(area)
        
        if not knowledge:
            print(f"⚠️  No knowledge found for {area}, skipping...")
            continue
        
        # Generate examples with different score combinations
        for i in range(num_examples_per_area):
            # Vary scores to create diverse examples
            base_score = np.random.uniform(0.2, 0.8)
            target_score = min(0.95, base_score + np.random.uniform(0.1, 0.3))
            
            # Create input with context
            all_scores = np.random.uniform(0.3, 0.9, 8)
            all_scores[areas.index(area)] = base_score  # Set the focus area score
            
            input_text = f"""Area: {area}
Current Score: {base_score:.2f}
Target Score: {target_score:.2f}
All Quality Scores: {[f'{s:.2f}' for s in all_scores]}
Swing Type: {np.random.choice(['power_swing', 'precision_swing', 'balanced_swing'])}"""
            
            # Generate response
            output = create_feedback_response(area, base_score, knowledge)
            
            training_data.append({
                "instruction": "Provide personalized golf swing feedback based on quality scores and golf instruction principles.",
                "input": input_text,
                "output": output
            })
    
    return training_data


def main():
    """Generate and save training data"""
    print("=" * 60)
    print("Generating Golf Feedback Training Data")
    print("=" * 60)
    
    # Generate training examples
    print("\nGenerating training examples from knowledge base...")
    training_data = generate_training_examples(num_examples_per_area=50)
    
    print(f"Generated {len(training_data)} training examples")
    
    # Split into train/val/test (85/10/5)
    train_portion = int(len(training_data) * 0.85)
    val_portion = int(len(training_data) * 0.10)
    
    train_data = training_data[:train_portion]
    val_data = training_data[train_portion:train_portion + val_portion]
    test_data = training_data[train_portion + val_portion:]
    
    print(f"\nSplit:")
    print(f"  Training: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    # Save all data
    output_dir = Path('experimental/golf_feedback_finetuning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    with open(output_dir / 'golf_feedback_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Save splits
    with open(output_dir / 'golf_feedback_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'golf_feedback_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'golf_feedback_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved training data to {output_dir}/")
    print(f"   - golf_feedback_training_data.json (full dataset)")
    print(f"   - golf_feedback_train.json")
    print(f"   - golf_feedback_val.json")
    print(f"   - golf_feedback_test.json")
    
    # Show example
    print("\n" + "=" * 60)
    print("Example Training Entry:")
    print("=" * 60)
    example = training_data[0]
    print(f"Instruction: {example['instruction']}")
    print(f"\nInput:\n{example['input']}")
    print(f"\nOutput (first 200 chars):\n{example['output'][:200]}...")


if __name__ == '__main__':
    main()
