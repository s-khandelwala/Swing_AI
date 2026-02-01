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
    ])
    .add_local_file("train.py", "/root/train.py")
    .add_local_file("dataloader.py", "/root/dataloader.py")
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("MobileNetV2.py", "/root/MobileNetV2.py")
    .add_local_file("util.py", "/root/util.py")
    .add_local_file("eval.py", "/root/eval.py")
    .add_local_file("mobilenet_v2.pth.tar", "/root/mobilenet_v2.pth.tar")
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

app = modal.App("golfdb-training")

# Create volumes for data and model checkpoints
videos_volume = modal.Volume.from_name("golfdb-videos", create_if_missing=True)
models_volume = modal.Volume.from_name("golfdb-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data/videos_160": videos_volume},
    timeout=60,
)
def list_uploaded_videos():
    """Get list of videos already in the volume"""
    import os
    
    volume_dir = "/data/videos_160"
    if os.path.exists(volume_dir):
        video_files = [f for f in os.listdir(volume_dir) if f.endswith('.mp4')]
        return set(video_files)  # Return as set for fast lookup
    return set()

@app.function(
    image=image,
    volumes={"/data/videos_160": videos_volume},
    timeout=3600,
)
def upload_batch_videos(video_batch: dict):
    """Upload multiple videos at once (faster - commits volume once per batch)"""
    import os
    
    volume_dir = "/data/videos_160"
    os.makedirs(volume_dir, exist_ok=True)
    
    results = []
    for video_name, video_data in video_batch.items():
        dest_path = os.path.join(volume_dir, video_name)
        if os.path.exists(dest_path):
            results.append({"status": "skipped", "name": video_name})
        else:
            with open(dest_path, 'wb') as f:
                f.write(video_data)
            results.append({"status": "uploaded", "name": video_name, "size": len(video_data)})
    
    # Commit once after all files in batch
    videos_volume.commit()
    return results


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data/videos_160": videos_volume,
        "/models": models_volume,
    },
    timeout=1800,
)
def evaluate(model_checkpoint="swingnet_2000.pth.tar", split=1):
    """Run evaluation on Modal GPU - with real-time output streaming"""
    import subprocess
    import sys
    import os
    
    print("=" * 60)
    print("Starting evaluation function...")
    print("=" * 60)
    
    os.chdir("/root")
    
    # Check model exists
    model_path = os.path.join("/models", model_checkpoint)
    if not os.path.exists(model_path):
        # Try copying from models volume if it's there
        print(f"Model {model_checkpoint} not found in /models")
        print("Available models:")
        if os.path.exists("/models"):
            models = [f for f in os.listdir("/models") if f.endswith('.pth.tar')]
            for m in models:
                print(f"  - {m}")
        return None
    
    # Copy model to local directory for eval.py
    import shutil
    local_model_path = f"models/{model_checkpoint}"
    os.makedirs("models", exist_ok=True)
    shutil.copy2(model_path, local_model_path)
    print(f"Using model: {model_checkpoint}")
    
    # Create a modified eval.py that uses the correct model path
    eval_script = f'''
from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds

def eval(model, split, seq_length, n_cpu, disp, vid_dir='/data/videos_160/'):
    dataset = GolfDB(data_file='data/val_split_{{}}.pkl'.format(split),
                     vid_dir=vid_dir,
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    device = next(model.parameters()).device

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)
    return PCE

if __name__ == '__main__':
    split = {split}
    seq_length = 64
    n_cpu = 6
    model_path = '{local_model_path}'

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load(model_path, map_location='cuda')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True)
    print('Average PCE: {{}}'.format(PCE))
'''
    
    # Write and run the eval script
    with open("run_eval.py", "w") as f:
        f.write(eval_script)
    
    print("Starting evaluation...")
    sys.stdout.flush()
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [sys.executable, "-u", "run_eval.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
        print(f"Error during evaluation: {e}")
        process.kill()
        return_code = 1
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed with return code: {return_code}")
    print(f"{'='*60}")
    
    return return_code

@app.local_entrypoint()
def eval_main(model_checkpoint: str = "swingnet_2000.pth.tar", split: int = 1):
    """Entry point - runs evaluation"""
    print(f"Starting evaluation on Modal GPU...")
    print(f"Model: {model_checkpoint}, Split: {split}")
    
    evaluate.remote(model_checkpoint, split)
    
    print("Evaluation complete!")


# Cost monitoring: Modal A100 40GB pricing = $0.000583/sec = $2.10/hour
# With $20 budget, we can afford ~9.5 hours max
# Setting timeout to 9.5 hours (34200 seconds) = ~$19.95 to stay safely under $20 budget
MAX_TRAINING_TIME_SECONDS = 34200  # 9.5 hours = ~$19.95 max (stays under $20 budget)

@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data/videos_160": videos_volume,
        "/models": models_volume,
    },
    timeout=MAX_TRAINING_TIME_SECONDS,
)
def train():
    """Run train.py on Modal GPU - with real-time output streaming"""
    import subprocess
    import sys
    import os
    
    print("=" * 60)
    print("Starting training function...")
    print("=" * 60)
    
    os.chdir("/root")
    os.makedirs("models", exist_ok=True)
    
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
    
    print("Starting train.py...")
    sys.stdout.flush()
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [sys.executable, "-u", "train.py"],  # -u for unbuffered
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
        process.kill()
        return_code = 1
    
    print(f"\n{'='*60}")
    print(f"Training completed with return code: {return_code}")
    print(f"{'='*60}")
    
    # Read training cost from file
    training_cost = None
    cost_file = "training_cost.txt"
    if os.path.exists(cost_file):
        try:
            with open(cost_file, 'r') as f:
                training_cost = float(f.read().strip())
            print(f"Training cost: ${training_cost:.2f}")
        except Exception as e:
            print(f"Warning: Could not read cost file: {e}")
            training_cost = None
    else:
        print("Warning: Cost file not found, cannot determine training cost")
    
    # Copy saved models to volume
    if os.path.exists("models"):
        print(f"Copying models to volume...")
        import shutil
        files_copied = 0
        for file in os.listdir("models"):
            shutil.copy2(f"models/{file}", f"/models/{file}")
            files_copied += 1
            print(f"  Copied: {file}")
        print(f"Copied {files_copied} model files to volume")
        models_volume.commit()
    else:
        print("No models directory found after training")
    
    # Return cost along with return code
    return {"return_code": return_code, "cost": training_cost}

@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def download_models():
    """List models in volume"""
    import os
    
    if os.path.exists("/models"):
        files = os.listdir("/models")
        print(f"Found {len(files)} model files in volume:")
        for file in files:
            file_path = os.path.join("/models", file)
            size = os.path.getsize(file_path)
            print(f"  {file} ({size / (1024*1024):.2f} MB)")
        return files
    else:
        print("No models directory found in volume")
        return []

@app.local_entrypoint()
def upload():
    """Upload videos to Modal volume - runs locally with batch uploading
    
    Set BATCH_SIZE environment variable to control batch size (default: 200).
    Modal supports up to 4 GiB per HTTP request. With ~5 MB videos, 
    you can safely use batch_size up to 700-800 videos per batch.
    
    Example: BATCH_SIZE=500 python -m modal run modal_runner.py::upload
    """
    import os
    
    # Get batch size from environment variable or use default
    batch_size = int(os.environ.get('BATCH_SIZE', '200'))
    
    # First, get list of videos already in volume
    print("Checking which videos are already uploaded...")
    uploaded_set = list_uploaded_videos.remote()
    print(f"Found {len(uploaded_set)} videos already in volume")
    
    local_video_dir = Path("data/videos_160")
    
    if not local_video_dir.exists():
        print(f"ERROR: Local video directory not found: {local_video_dir}")
        return
    
    # Get all local videos
    all_video_files = list(local_video_dir.glob("*.mp4"))
    print(f"Found {len(all_video_files)} videos locally")
    
    # Filter out videos that already exist
    video_files = [v for v in all_video_files if v.name not in uploaded_set]
    print(f"Need to upload {len(video_files)} videos (skipping {len(all_video_files) - len(video_files)} already uploaded)")
    
    if len(video_files) == 0:
        print("All videos already uploaded!")
        return
    
    # Warn if batch size might exceed limits
    if batch_size > 800:
        print(f"Warning: batch_size {batch_size} is very large. Modal supports up to 4 GiB per request.")
        print("If videos average >5 MB each, consider reducing batch_size to avoid errors.")
    
    print(f"Using batch size: {batch_size} videos per batch")
    
    # Process in batches
    uploaded = 0
    failed = 0
    
    def process_batch(batch):
        """Process a single batch of videos"""
        batch_dict = {}
        total_size = 0
        for video_path in batch:
            try:
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                    batch_dict[video_path.name] = video_data
                    total_size += len(video_data)
            except Exception as e:
                print(f"  ERROR: Error reading {video_path.name}: {e}")
                return None, 0
        return batch_dict, total_size
    
    print(f"\nUploading in batches of {batch_size} videos...")
    
    # Process batches sequentially (but each batch uploads all videos at once)
    total_batches = (len(video_files) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(video_files))
        batch = video_files[start_idx:end_idx]
        
        print(f"\nBatch {batch_num + 1}/{total_batches} ({len(batch)} videos)...")
        
        # Read all files in batch
        batch_dict, batch_size_bytes = process_batch(batch)
        if batch_dict is None:
            failed += len(batch)
            continue
        
        batch_size_mb = batch_size_bytes / (1024 * 1024)
        batch_size_gb = batch_size_bytes / (1024 * 1024 * 1024)
        print(f"  Batch size: {batch_size_mb:.2f} MB ({batch_size_gb:.3f} GiB)")
        
        # Check if batch exceeds Modal's 4 GiB limit
        if batch_size_gb > 3.5:
            print(f"  WARNING: Batch size ({batch_size_gb:.3f} GiB) is close to Modal's 4 GiB limit!")
        
        # Upload batch
        try:
            results = upload_batch_videos.remote(batch_dict)
            batch_uploaded = sum(1 for r in results if r["status"] == "uploaded")
            batch_skipped = sum(1 for r in results if r["status"] == "skipped")
            uploaded += batch_uploaded
            
            if batch_uploaded > 0:
                print(f"  OK: Uploaded {batch_uploaded} videos in this batch")
            if batch_skipped > 0:
                print(f"  SKIP: Skipped {batch_skipped} videos (already exist)")
            
            if uploaded % 100 == 0:  # Print progress every 100 videos
                print(f"  Progress: {uploaded} total videos uploaded so far...")
        except Exception as e:
            failed += len(batch)
            print(f"  ERROR: Error uploading batch: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Upload complete! {uploaded} new videos uploaded, {failed} failed.")
    print(f"Total videos in volume: {len(uploaded_set) + uploaded}")
    print(f"{'='*60}")

@app.local_entrypoint()
def main():
    """Entry point - runs training, then evaluation if under budget"""
    print("Starting training on Modal GPU...")
    print("Note: Videos should be in volume 'golfdb-videos'")
    
    # Run training
    result = train.remote()
    
    # Handle result (can be dict with cost, or just return code for backwards compatibility)
    if isinstance(result, dict):
        return_code = result.get("return_code", 0)
        training_cost = result.get("cost", None)
    else:
        return_code = result if result is not None else 0
        training_cost = None
    
    print("\n" + "="*60)
    print("Training complete!")
    print("Model checkpoints saved to volume 'golfdb-models'")
    if training_cost is not None:
        print(f"Training cost: ${training_cost:.2f}")
    
    # Check if we can afford evaluation (stay under $23 total)
    EVAL_BUDGET_LIMIT = 23.0
    if training_cost is not None and training_cost < EVAL_BUDGET_LIMIT:
        remaining_budget = EVAL_BUDGET_LIMIT - training_cost
        print(f"\nBudget remaining for evaluation: ${remaining_budget:.2f}")
        print(f"Running evaluation on best model (swingnet_best.pth.tar)...")
        print("="*60 + "\n")
        
        evaluate.remote("swingnet_best.pth.tar", split=1)
        
        print("\n" + "="*60)
        print("Evaluation complete!")
        print("="*60)
    elif training_cost is not None:
        print(f"\nSkipping evaluation: Training cost (${training_cost:.2f}) exceeds budget limit (${EVAL_BUDGET_LIMIT:.2f})")
        print("="*60)
    else:
        print("\nSkipping evaluation: Could not determine training cost")
        print("="*60)

@app.local_entrypoint()
def download():
    """Download models list"""
    files = download_models.remote()
    if files:
        print(f"\nTo download models, use Modal web dashboard or CLI")
        print(f"Volume: golfdb-models")

@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def download_model(model_name: str):
    """Download a specific model from Modal volume"""
    import os
    
    model_path = os.path.join("/models", model_name)
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found in volume")
        print("Available models:")
        if os.path.exists("/models"):
            models = [f for f in os.listdir("/models") if f.endswith('.pth.tar')]
            for m in models:
                print(f"  - {m}")
        return None
    
    # Read the model file
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    return model_data

@app.local_entrypoint()
def download_model_local(model_name: str = "swingnet_2000.pth.tar", output_path: str = "models/swingnet_2000.pth.tar"):
    """Download a model from Modal volume to local machine"""
    import os
    from pathlib import Path
    
    print(f"Downloading {model_name} from Modal volume...")
    model_data = download_model.remote(model_name)
    
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