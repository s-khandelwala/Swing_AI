import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True, padding_after_finish=100):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.padding_after_finish = padding_after_finish  # Frames after Finish to include in training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

        images, labels = [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

        if self.train:
            # random starting position, sample 'seq_length' frames
            # Allow sampling from frame 0 to (Finish + padding_after_finish)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finish_frame = events[-1]  # Finish event frame (after adjustment)
            
            # Calculate max start frame: ensure we can sample seq_length frames
            # and include padding after Finish
            max_start = min(total_frames - self.seq_length, finish_frame + self.padding_after_finish)
            max_start = max(0, max_start)  # Ensure non-negative
            
            # If max_start is less than finish_frame, fall back to original behavior
            if max_start < finish_frame:
                start_frame = np.random.randint(finish_frame + 1)
            else:
                start_frame = np.random.randint(max_start + 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for pos in range(total_frames):
                ret, img = cap.read()
                if ret:  # Only append if frame was successfully read
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()
            
            # Validate that we got at least some frames
            if len(images) == 0:
                raise ValueError(f"Video {a['id']} has no valid frames")

        sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        
        # Validate array shape before transform
        if len(sample['images'].shape) != 4 or sample['images'].shape[0] == 0:
            raise ValueError(f"Video {a['id']} has invalid image array shape: {sample['images'].shape}")
        
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        
        # Validate shape before transpose (expecting (N, H, W, C))
        if len(images.shape) != 4 or images.shape[3] != 3:
            raise ValueError(f"ToTensor: Expected 4D array (N, H, W, C) with C=3, got shape {images.shape}")
        
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


class ColorJitter(object):
    """Color jitter augmentation for video sequences.
    
    Applies random brightness, contrast, saturation, and hue adjustments.
    Each frame in the sequence gets the same augmentation parameters.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05):
        from torchvision import transforms as T
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        # images shape: (N, C, H, W) - sequence of frames
        # ColorJitter applies random transformations to each frame
        # Note: For video, ideally same jitter per sequence, but this still helps
        jittered = self.color_jitter(images)
        
        return {'images': jittered, 'labels': labels}


class RandomErasing(object):
    """Random Erasing Data Augmentation for video sequences.
    
    Paper: https://arxiv.org/abs/1708.04896
    Randomly erase rectangular regions from frames.
    Works with tensor format: (N, C, H, W)
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        
        if np.random.rand() > self.probability:
            return {'images': images, 'labels': labels}
        
        # images shape: (N, C, H, W) - tensor
        N, C, H, W = images.shape
        
        # Apply random erasing to random frames in sequence
        num_erase = max(1, int(N * 0.3))  # Erase ~30% of frames
        erase_indices = np.random.choice(N, size=num_erase, replace=False)
        
        for t in erase_indices:
            area = H * W
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1/self.r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < W and h < H:
                x1 = np.random.randint(0, H - h)
                y1 = np.random.randint(0, W - w)
                
                if C == 3:
                    images[t, :, x1:x1+h, y1:y1+w] = self.mean.to(images.device)
                else:
                    images[t, :, x1:x1+h, y1:y1+w] = self.mean[0, 0, 0, 0].to(images.device)
        
        return {'images': images, 'labels': labels}


if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=64,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))




    





       

