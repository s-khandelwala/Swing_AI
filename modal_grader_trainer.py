"""
Modal script for training the Golf Swing Quality Grader

This script runs the golf swing grader training on Modal's GPU infrastructure.
Based on professional golf instruction principles (Golf Digest, MyTPI, Hackmotion).
"""

import modal
import os
from pathlib import Path

# Image with dependencies and code files
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    ])
    .pip_install([
        "torch>=1.0.0",
        "torchvision>=0.2.0",
        "opencv-python>=4.0.0",
        "numpy>=1.15.0",
        "scipy>=1.0.0",
        "pandas>=0.23.0",
        "tqdm>=4.60.0",  # For progress bars
    ])
    # Core model files
    .add_local_file("golf_swing_grader.py", "/root/golf_swing_grader.py")
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("dataloader.py", "/root/dataloader.py")
    .add_local_file("MobileNetV2.py", "/root/MobileNetV2.py")
    .add_local_file("util.py", "/root/util.py")
    .add_local_file("mobilenet_v2.pth.tar", "/root/mobilenet_v2.pth.tar")
    # Data files
    .add_local_file("data/golfDB.pkl", "/root/data/golfDB.pkl")
    .add_local_file("data/train_split_1.pkl", "/root/data/train_split_1.pkl")
    .add_local_file("data/train_split_2.pkl", "/root/data/train_split_2.pkl")
    .add_local_file("data/train_split_3.pkl", "/root/data/train_split_3.pkl")
    .add_local_file("data/train_split_4.pkl", "/root/data/train_split_4.pkl")
    .add_local_file("data/val_split_1.pkl", "/root/data/val_split_1.pkl")
    .add_local_file("data/val_split_2.pkl", "/root/data/val_split_2.pkl")
    .add_local_file("data/val_split_3.pkl", "/root/data/val_split_3.pkl")
    .add_local_file("data/val_split_4.pkl", "/root/data/val_split_4.pkl")
)

app = modal.App("golf-swing-grader-training")

# Create volumes for data and model checkpoints
videos_volume = modal.Volume.from_name("golfdb-videos", create_if_missing=True)
models_volume = modal.Volume.from_name("golfdb-models", create_if_missing=True)

# Cost monitoring: Modal A100 40GB pricing = $0.000583/sec = $2.10/hour
# Setting timeout to 9.5 hours (34200 seconds) = ~$19.95 max
MAX_TRAINING_TIME_SECONDS = 34200  # 9.5 hours

@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data/videos_160": videos_volume,
        "/models": models_volume,
    },
    timeout=MAX_TRAINING_TIME_SECONDS,
)
def train_grader(
    event_detector_path="swingnet_best.pth.tar",
    data_file="train_split_1.pkl",
    epochs=50,
    batch_size=16,
    lr=0.001,
    seq_length=64,
    videos_normalized=False
):
    """
    Train Golf Swing Quality Grader on Modal GPU.
    
    Args:
        event_detector_path: Path to pretrained EventDetector model (in /models volume)
        data_file: Training data split file (e.g., "train_split_1.pkl")
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        seq_length: Sequence length for video frames
        videos_normalized: Whether videos have been normalized (affects frame quality signal)
    """
    import subprocess
    import sys
    import os
    
    print("=" * 60)
    print("Golf Swing Quality Grader Training")
    print("Based on Golf Digest, MyTPI, Hackmotion principles")
    print("=" * 60)
    
    os.chdir("/root")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Check video directory
    video_dir = "/data/videos_160"
    if os.path.exists(video_dir):
        video_count = len([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        print(f"Found {video_count} videos in {video_dir}")
        if video_count == 0:
            print("ERROR: No videos found! Run upload_videos first.")
            return {"return_code": 1, "cost": None}
    else:
        print(f"WARNING: Video directory {video_dir} not found!")
        return {"return_code": 1, "cost": None}
    
    # Check if event detector model exists
    event_detector_full_path = f"/models/{event_detector_path}"
    if not os.path.exists(event_detector_full_path):
        print(f"ERROR: Event detector model not found: {event_detector_full_path}")
        print("Available models in /models:")
        if os.path.exists("/models"):
            models = [f for f in os.listdir("/models") if f.endswith('.pth.tar')]
            for m in models:
                print(f"  - {m}")
        return {"return_code": 1, "cost": None}
    
    # Copy event detector to local directory for training script
    import shutil
    local_event_detector = f"models/{event_detector_path}"
    shutil.copy2(event_detector_full_path, local_event_detector)
    print(f"Using event detector: {event_detector_path}")
    
    # Create training script
    training_script = f'''
import sys
import os
sys.path.insert(0, "/root")

from golf_swing_grader import train_golf_swing_grader

# Training configuration
train_golf_swing_grader(
    event_detector_path="models/{event_detector_path}",
    data_file="data/{data_file}",
    vid_dir="/data/videos_160/",
    epochs={epochs},
    batch_size={batch_size},
    lr={lr},
    seq_length={seq_length},
    videos_normalized={str(videos_normalized).lower()}
)
'''
    
    # Write training script
    with open("train_grader.py", "w") as f:
        f.write(training_script)
    
    print("\n" + "=" * 60)
    print("Starting grader training...")
    print(f"Configuration:")
    print(f"  Event Detector: {event_detector_path}")
    print(f"  Data File: {data_file}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Videos Normalized: {videos_normalized}")
    print("=" * 60 + "\n")
    
    sys.stdout.flush()
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [sys.executable, "-u", "train_grader.py"],  # -u for unbuffered
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Stream output line by line
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                sys.stdout.flush()
        process.stdout.close()
        return_code = process.wait()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        process.kill()
        return_code = 1
    
    print(f"\n{'='*60}")
    print(f"Training completed with return code: {return_code}")
    print(f"{'='*60}")
    
    # Copy saved models to volume
    if os.path.exists("models"):
        print(f"\nCopying models to volume...")
        files_copied = 0
        for file in os.listdir("models"):
            if file.startswith("golf_swing_grader") or file.startswith("golf_instruction_labels"):
                src_path = f"models/{file}"
                dest_path = f"/models/{file}"
                shutil.copy2(src_path, dest_path)
                files_copied += 1
                print(f"  Copied: {file}")
        print(f"Copied {files_copied} model files to volume")
        models_volume.commit()
    else:
        print("No models directory found after training")
    
    # Also copy labels if generated
    if os.path.exists("data/golf_instruction_labels.pkl"):
        print("Copying training labels to volume...")
        shutil.copy2("data/golf_instruction_labels.pkl", "/models/golf_instruction_labels.pkl")
        models_volume.commit()
    
    return {"return_code": return_code}


@app.local_entrypoint()
def main(
    event_detector: str = "swingnet_best.pth.tar",
    data_file: str = "train_split_1.pkl",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    videos_normalized: bool = False
):
    """
    Entry point for training golf swing grader.
    
    Args:
        event_detector: Name of event detector model in volume (default: swingnet_best.pth.tar)
        data_file: Training data split (default: train_split_1.pkl)
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size (default: 16)
        lr: Learning rate (default: 0.001)
        videos_normalized: Whether videos are normalized (default: False)
    """
    print("=" * 60)
    print("Golf Swing Quality Grader Training on Modal")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Event Detector: {event_detector}")
    print(f"  Data File: {data_file}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Videos Normalized: {videos_normalized}")
    print("\nStarting training on Modal GPU...")
    print("=" * 60 + "\n")
    
    result = train_grader.remote(
        event_detector_path=event_detector,
        data_file=data_file,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        videos_normalized=videos_normalized
    )
    
    print("\n" + "=" * 60)
    if isinstance(result, dict):
        return_code = result.get("return_code", 0)
        if return_code == 0:
            print("Training completed successfully!")
            print("Model checkpoints saved to volume 'golfdb-models'")
        else:
            print(f"Training failed with return code: {return_code}")
    else:
        print("Training completed!")
    print("=" * 60)


@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def list_grader_models():
    """List all grader models in the volume"""
    import os
    
    if os.path.exists("/models"):
        files = os.listdir("/models")
        grader_models = [f for f in files if "golf_swing_grader" in f or "golf_instruction" in f]
        print(f"Found {len(grader_models)} grader-related files:")
        for file in sorted(grader_models):
            file_path = os.path.join("/models", file)
            size = os.path.getsize(file_path)
            print(f"  {file} ({size / (1024*1024):.2f} MB)")
        return grader_models
    else:
        print("No models directory found in volume")
        return []


@app.local_entrypoint()
def list_models():
    """List available grader models"""
    models = list_grader_models.remote()
    if models:
        print(f"\nTo download models, use Modal web dashboard or CLI")
        print(f"Volume: golfdb-models")


@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def download_grader_model(model_name: str):
    """Download a specific grader model from Modal volume"""
    import os
    
    model_path = os.path.join("/models", model_name)
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found in volume")
        print("Available grader models:")
        list_grader_models.remote()
        return None
    
    # Read the model file
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    return model_data


@app.local_entrypoint()
def download_model_local(
    model_name: str = "golf_swing_grader_final.pth.tar",
    output_path: str = "models/golf_swing_grader_final.pth.tar"
):
    """Download a grader model from Modal volume to local machine"""
    import os
    from pathlib import Path
    
    print(f"Downloading {model_name} from Modal volume...")
    model_data = download_grader_model.remote(model_name)
    
    if model_data is None:
        print("Download failed!")
        return
    
    # Create models directory if it doesn't exist
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Write model to local file
    with open(output_path, 'wb') as f:
        f.write(model_data)
    
    size_mb = len(model_data) / (1024 * 1024)
    print(f"OK: Downloaded {model_name} ({size_mb:.2f} MB) to {output_path}")


if __name__ == "__main__":
    # Default training run
    main()

