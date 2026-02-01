from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds


def safe_collate_fn(batch):
    """Custom collate function that handles None values (from failed video loads)"""
    # Filter out None values (failed loads)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # Use default collate for valid samples (batch_size=1, so just return first item)
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


class SafeDataset(torch.utils.data.Dataset):
    """Wrapper that catches errors and returns None for problematic videos"""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception:
            # Return None for failed videos - will be filtered by collate_fn
            return None


def eval(model, split, seq_length, n_cpu, disp, vid_dir='data/videos_160/'):
    base_dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                          vid_dir=vid_dir,
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)
    dataset = SafeDataset(base_dataset)

    # Use parallel workers for faster data loading (videos can't be batched due to different lengths)
    num_workers = min(4, n_cpu)  # Use up to 4 workers for parallel data loading
    data_loader = DataLoader(dataset,
                             batch_size=1,  # Must be 1 - videos have different lengths, can't batch
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=False,
                             collate_fn=safe_collate_fn)

    correct = []
    skipped = []
    # Get the device from the model
    device = next(model.parameters()).device
    
    # Get total number of videos for progress tracking
    total_videos = len(dataset)
    print(f'  Evaluating {total_videos} videos (using {num_workers} parallel workers)...', flush=True)
    import time
    start_time = time.time()

    for i, sample in enumerate(data_loader):
        # Skip None samples (from safe_collate_fn when all videos in batch failed)
        if sample is None:
            print(f'  Warning: Skipping video {i} - error during loading', flush=True)
            skipped.append(i)
            continue
            
        try:
            images, labels = sample['images'], sample['labels']
            
            # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                logits = model(image_batch.to(device))  # Use GPU device instead of CPU
                if batch == 0:
                    probs = F.softmax(logits.data, dim=1).cpu().numpy()
                else:
                    probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
                batch += 1
            _, _, _, _, c = correct_preds(probs, labels.squeeze())
            if disp:
                print(i, c)
            # c is an array of shape (8,) indicating which events were correct
            # We want the mean correctness across all 8 events for this video
            correct.append(np.mean(c))
        except Exception as e:
            print(f'  Warning: Skipping video {i} due to error: {str(e)[:100]}', flush=True)
            skipped.append(i)
            continue
        
        # Print progress every 50 videos
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_videos - (i + 1)) / rate if rate > 0 else 0
            print(f'  Progress: {i + 1}/{total_videos} videos ({100*(i+1)/total_videos:.1f}%) - '
                  f'Elapsed: {elapsed/60:.1f}m, Est. remaining: {remaining/60:.1f}m', flush=True)
    
    elapsed_total = time.time() - start_time
    print(f'  Validation complete: {len(correct)}/{total_videos} videos processed in {elapsed_total/60:.1f} minutes', flush=True)
    if skipped:
        print(f'  Skipped {len(skipped)} problematic videos: {skipped[:10]}{"..." if len(skipped) > 10 else ""}', flush=True)
    if len(correct) == 0:
        print('  ERROR: No videos processed successfully!', flush=True)
        return 0.0
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    split = 1
    seq_length = 64
    n_cpu = 6
    model_path = 'models/swingnet_1800.pth.tar'  # Can be changed via command line

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load(model_path, map_location='cuda')  # Load on GPU
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()  # Move model to GPU
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True)
    print('Average PCE: {}'.format(PCE))


