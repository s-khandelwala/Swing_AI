"""
Training utility functions for optimizations:
- Focal Loss (for class imbalance)
- Exponential Moving Average (EMA)
- Stochastic Weight Averaging (SWA)
- MixUp augmentation
- Random Erasing augmentation
- Weight initialization
- Lookahead Optimizer
- Learning Rate Warmup
- Huber Loss
- Learning Rate Finder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from collections import defaultdict
import numpy as np
import copy


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EMA:
    """Exponential Moving Average for model weights.
    
    Better final model weights by averaging over training trajectory.
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with exponential moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation for data.
    
    Paper: https://arxiv.org/abs/1710.09412
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp criterion combining losses from both labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def init_weights(m):
    """Initialize model weights with Xavier uniform initialization."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps LSTM remember)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Lookahead(Optimizer):
    """Lookahead Optimizer Wrapper.
    
    Paper: https://arxiv.org/abs/1907.08610
    Implements lookahead algorithm on top of base optimizer.
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss
    
    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }
    
    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class SWA:
    """Stochastic Weight Averaging.
    
    Paper: https://arxiv.org/abs/1803.05407
    Averages model weights during the final training phase for better generalization.
    """
    def __init__(self, model, swa_start=None, swa_freq=None, swa_lr=None):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.swa_n = 0
        self.swa_model = None
        self.swa_scheduler = None
        
    def update_swa(self):
        """Update SWA model with current model weights."""
        if self.swa_model is None:
            self.swa_model = copy.deepcopy(self.model)
            self.swa_n = 1
        else:
            for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
                swa_param.data = (swa_param.data * self.swa_n + param.data) / (self.swa_n + 1)
            self.swa_n += 1
    
    def get_swa_model(self):
        """Return SWA model."""
        return self.swa_model
    
    def apply_swa(self):
        """Apply SWA weights to model."""
        if self.swa_model is not None:
            self.model.load_state_dict(self.swa_model.state_dict())


class RandomErasing:
    """Random Erasing Data Augmentation.
    
    Paper: https://arxiv.org/abs/1708.04896
    Randomly erase rectangular regions from images.
    Works with dataloader sample format: {'images': (T, C, H, W), 'labels': ...}
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        
    def __call__(self, sample):
        if 'images' not in sample:
            return sample
        
        if np.random.rand() > self.probability:
            return sample
        
        images = sample['images']  # Shape: (T, C, H, W) - numpy array
        T, C, H, W = images.shape
        
        # Apply random erasing to random frames in sequence
        num_erase = max(1, int(T * 0.3))  # Erase ~30% of frames
        erase_indices = np.random.choice(T, size=num_erase, replace=False)
        
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
                    images[t, 0, x1:x1+h, y1:y1+w] = self.mean[0]
                    images[t, 1, x1:x1+h, y1:y1+w] = self.mean[1]
                    images[t, 2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    images[t, :, x1:x1+h, y1:y1+w] = self.mean[0]
        
        sample['images'] = images
        return sample


class WarmupScheduler(_LRScheduler):
    """Learning Rate Warmup Scheduler Wrapper.
    
    Wraps an existing scheduler and adds warmup phase.
    Gradually increases learning rate from 0 to initial LR over warmup steps.
    """
    def __init__(self, scheduler, warmup_steps, last_epoch=-1):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]
        super(WarmupScheduler, self).__init__(scheduler.optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            # Use wrapped scheduler
            return self.scheduler.get_lr()
    
    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super(WarmupScheduler, self).step(epoch)
        else:
            self.scheduler.step(epoch)
        self.last_epoch += 1


class HuberLoss(nn.Module):
    """Huber Loss (Smooth L1 with configurable delta).
    
    Less sensitive to outliers than MSE, more smooth than L1.
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def find_learning_rate(model, train_loader, criterion, optimizer, device, init_lr=1e-8, final_lr=1.0, beta=0.98):
    """Learning Rate Finder.
    
    Finds optimal learning rate by exponentially increasing LR and tracking loss.
    Adapted from: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    num_iter = len(train_loader)
    mult = (final_lr / init_lr) ** (1 / num_iter)
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    
    model.train()
    for data in train_loader:
        batch_num += 1
        # Get data (adjust based on your data loader)
        if isinstance(data, dict):
            images = data['images'].to(device)
            labels = data['labels'].to(device)
        else:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle different output shapes
        if len(outputs.shape) > 2:
            outputs = outputs.view(-1, outputs.size(-1))
        if len(labels.shape) > 1:
            labels = labels.view(-1)
        
        loss = criterion(outputs, labels)
        
        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        
        # Stop if loss explodes
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    
    return log_lrs, losses

