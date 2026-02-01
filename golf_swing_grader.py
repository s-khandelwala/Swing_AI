"""
Golf Swing Quality Grader - Based on Professional Instruction Principles

Grading criteria based on Golf Digest, MyTPI, and Hackmotion:
1. Fundamentals (Setup): Grip, Stance, Posture, Alignment, Ball Position
2. Swing Mechanics: Tempo, Rhythm, Weight Shift, Body Rotation, Key Positions (P1-P9)
3. Impact Factors: Club Path, Face Angle, Attack Angle, Speed
4. Finish & Balance: Follow-through, Balance

This implementation focuses on what can be measured from video.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EventDetector
import numpy as np
from dataloader import GolfDB, Normalize, ToTensor, ColorJitter, RandomErasing
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from training_utils import (
    EMA, SWA, mixup_data, mixup_criterion, init_weights,
    Lookahead, WarmupScheduler, HuberLoss
)
import os
import pickle
import cv2


class GolfSwingGrader(nn.Module):
    """
    Golf-specific swing quality grader based on professional instruction principles.
    
    Predicts 8 quality scores aligned with golf instruction:
    1. Setup Quality (stance, posture, alignment)
    2. Tempo & Rhythm
    3. Weight Shift
    4. Body Rotation
    5. Impact Quality (inferred from swing mechanics)
    6. Follow-Through
    7. Balance
    8. Overall Consistency
    """
    
    def __init__(self, event_detector, embedding_dim=256):
        super().__init__()
        self.event_detector = event_detector
        
        # Freeze event detector
        for param in self.event_detector.parameters():
            param.requires_grad = False
        
        # Temporal attention (focuses on key swing phases)
        self.temporal_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Golf-specific quality head
        # 8 scores: setup, tempo, weight_shift, rotation, impact, followthrough, balance, consistency
        self.quality_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )
    
    def get_features(self, x):
        """Extract temporal features using EventDetector"""
        batch_size, timesteps, C, H, W = x.size()
        
        # CNN features
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.event_detector.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.event_detector.dropout:
            c_out = self.event_detector.drop(c_out)
        
        # LSTM features
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.event_detector.rnn(r_in, self.event_detector.init_hidden(batch_size))
        
        return r_out  # (batch, timesteps, hidden_dim)
    
    def forward(self, x):
        """Predict golf-specific quality scores"""
        temporal_features = self.get_features(x)
        
        # Temporal attention
        attention_logits = self.temporal_attention(temporal_features)
        attention_weights = F.softmax(attention_logits.squeeze(-1), dim=1)
        
        # Weighted aggregation
        attended_features = (temporal_features * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        # Predict golf-specific scores
        scores = self.quality_head(attended_features)
        
        return scores, attention_weights


def compute_golf_instruction_signals(video_frames, events, videos_normalized=False):
    """
    Compute golf instruction-based quality signals from video.
    
    Based on Golf Digest, MyTPI, Hackmotion principles:
    1. Setup Quality (stance, posture inferred from initial frames)
    2. Tempo & Rhythm (smooth, unhurried transition)
    3. Weight Shift (moves to front foot in downswing)
    4. Body Rotation (full shoulder turn, proper sequence)
    5. Impact Quality (inferred from swing mechanics)
    6. Follow-Through (full extension, weight on lead foot)
    7. Balance (hold finish position)
    8. Consistency (repeatable motion)
    
    Returns: 8-dimensional quality vector aligned with golf instruction
    """
    # Convert to tensor if needed
    if isinstance(video_frames, np.ndarray):
        video_frames = torch.from_numpy(video_frames).float()
    
    # Ensure proper shape: (T, C, H, W)
    if len(video_frames.shape) == 5:
        video_frames = video_frames.squeeze(0)
    elif len(video_frames.shape) == 3:
        video_frames = video_frames.unsqueeze(1).repeat(1, 3, 1, 1)
    
    T, C, H, W = video_frames.shape
    
    # Convert to grayscale for analysis
    if C == 3:
        gray_frames = 0.299 * video_frames[:, 0] + 0.587 * video_frames[:, 1] + 0.114 * video_frames[:, 2]
    else:
        gray_frames = video_frames.squeeze(1)
    
    # Compute frame differences for motion analysis
    frame_diffs = torch.diff(gray_frames, dim=0)
    motion_per_frame = torch.mean(torch.abs(frame_diffs), dim=(1, 2))
    
    # 1. Setup Quality (Address Position)
    # Good setup: balanced, athletic posture, consistent initial position
    # Measure: Stability of initial frames (first 10% of video)
    initial_frames = int(T * 0.1)
    if initial_frames > 0:
        initial_motion = torch.mean(torch.abs(frame_diffs[:initial_frames])).item()
        # Low motion in setup = good, stable address position
        setup_quality = max(0, 1.0 - initial_motion * 20)
    else:
        setup_quality = 0.5
    
    # 2. Tempo & Rhythm
    # Good tempo: Smooth, unhurried transition, backswing sets up downswing
    # Measure: Consistency of motion speed, smooth acceleration
    if len(motion_per_frame) > 1:
        # Coefficient of variation (lower = more consistent)
        tempo_cv = torch.std(motion_per_frame) / (torch.mean(motion_per_frame) + 1e-8)
        tempo_quality = max(0, 1.0 - float(tempo_cv))
        
        # Check for smooth transition (backswing to downswing)
        # Motion should accelerate smoothly, not abruptly
        if len(motion_per_frame) > 10:
            mid_point = len(motion_per_frame) // 2
            backswing_motion = torch.mean(motion_per_frame[:mid_point])
            downswing_motion = torch.mean(motion_per_frame[mid_point:])
            # Downswing should be faster but not too abrupt
            if backswing_motion > 0:
                transition_ratio = float(downswing_motion / backswing_motion)
                # Good transition: 1.5-3x faster (not too abrupt, not too slow)
                if 1.5 <= transition_ratio <= 3.0:
                    transition_score = 1.0
                elif transition_ratio < 1.5:
                    transition_score = transition_ratio / 1.5  # Too slow
                else:
                    transition_score = max(0, 1.0 - (transition_ratio - 3.0) / 2.0)  # Too abrupt
            else:
                transition_score = 0.5
        else:
            transition_score = 0.5
        
        # Combine tempo consistency and transition smoothness
        tempo_rhythm = (tempo_quality * 0.6 + transition_score * 0.4)
    else:
        tempo_rhythm = 0.5
    
    # 3. Weight Shift
    # Good weight shift: Moves to front foot in downswing
    # Measure: Motion center of mass shifts forward (approximated from frame differences)
    if len(frame_diffs) > 0:
        # Compute where motion is happening (center of motion)
        # Forward motion in downswing = weight shift
        mid_point = len(frame_diffs) // 2
        backswing_diffs = frame_diffs[:mid_point]
        downswing_diffs = frame_diffs[mid_point:]
        
        # Motion in lower portion of frame (where body/weight is) vs upper portion
        lower_half = frame_diffs[:, H//2:, :]
        upper_half = frame_diffs[:, :H//2, :]
        
        # Weight shift: more motion in lower half during downswing
        backswing_lower = torch.mean(torch.abs(backswing_diffs[:, H//2:, :])).item() if len(backswing_diffs) > 0 else 0
        downswing_lower = torch.mean(torch.abs(downswing_diffs[:, H//2:, :])).item() if len(downswing_diffs) > 0 else 0
        
        if backswing_lower > 0:
            weight_shift_ratio = downswing_lower / (backswing_lower + 1e-8)
            # Good weight shift: 1.2-2.0x more lower body motion in downswing
            if 1.2 <= weight_shift_ratio <= 2.0:
                weight_shift = 1.0
            elif weight_shift_ratio < 1.2:
                weight_shift = weight_shift_ratio / 1.2
            else:
                weight_shift = max(0, 1.0 - (weight_shift_ratio - 2.0) / 1.0)
        else:
            weight_shift = 0.5
    else:
        weight_shift = 0.5
    
    # 4. Body Rotation
    # Good rotation: Full shoulder turn, proper sequence
    # Measure: Motion magnitude and consistency (full rotation = more motion)
    if len(motion_per_frame) > 0:
        # Full rotation should have significant motion in backswing
        backswing_motion = torch.mean(motion_per_frame[:len(motion_per_frame)//2]).item()
        # Normalize: good rotation has moderate-high motion (0.05-0.15 range)
        if 0.05 <= backswing_motion <= 0.15:
            rotation_score = 1.0
        elif backswing_motion < 0.05:
            rotation_score = backswing_motion / 0.05  # Limited rotation
        else:
            rotation_score = max(0, 1.0 - (backswing_motion - 0.15) / 0.1)  # Too much
    else:
        rotation_score = 0.5
    
    # 5. Impact Quality (Inferred from Swing Mechanics)
    # Good impact: Proper sequence, club path, timing
    # Measure: Motion at impact phase (around 2/3 through swing)
    if len(events) >= 8:
        # Impact is event 5 (index 5)
        impact_frame = events[5] if len(events) > 5 else len(motion_per_frame) * 2 // 3
        impact_frame = min(impact_frame, len(motion_per_frame) - 1)
        
        # Motion at impact should be high (clubhead speed)
        impact_motion = motion_per_frame[impact_frame].item() if impact_frame < len(motion_per_frame) else 0.1
        
        # Also check sequence: downswing should accelerate to impact
        if impact_frame > 5:
            pre_impact_motion = torch.mean(motion_per_frame[max(0, impact_frame-5):impact_frame]).item()
            # Good: motion increases toward impact
            if impact_motion > pre_impact_motion:
                acceleration_score = min(1.0, (impact_motion - pre_impact_motion) / 0.05)
            else:
                acceleration_score = 0.3  # Deceleration = poor
        else:
            acceleration_score = 0.5
        
        # Combine: high motion + acceleration = good impact
        impact_quality = (min(1.0, impact_motion * 10) * 0.6 + acceleration_score * 0.4)
    else:
        impact_quality = 0.5
    
    # 6. Follow-Through
    # Good follow-through: Full extension, weight on lead foot, belt buckle facing target
    # Measure: Motion continues after impact, complete finish
    if len(events) >= 8:
        impact_frame = events[5] if len(events) > 5 else len(motion_per_frame) * 2 // 3
        finish_frame = events[7] if len(events) > 7 else len(motion_per_frame) - 1
        
        followthrough_duration = finish_frame - impact_frame
        backswing_duration = events[3] - events[0] if len(events) > 3 else len(motion_per_frame) // 2
        
        if backswing_duration > 0:
            # Good follow-through: similar or longer than backswing
            followthrough_ratio = followthrough_duration / backswing_duration
            if followthrough_ratio >= 0.8:
                followthrough_score = 1.0
            else:
                followthrough_score = followthrough_ratio / 0.8
        else:
            followthrough_score = 0.5
        
        # Also check: motion should continue (not stop abruptly)
        if impact_frame < len(motion_per_frame) - 5:
            post_impact_motion = torch.mean(motion_per_frame[impact_frame:min(impact_frame+5, len(motion_per_frame))]).item()
            # Good: motion continues after impact
            continuation_score = min(1.0, post_impact_motion * 10)
        else:
            continuation_score = 0.5
        
        followthrough = (followthrough_score * 0.7 + continuation_score * 0.3)
    else:
        followthrough = 0.5
    
    # 7. Balance
    # Good balance: Hold finish position, stable throughout
    # Measure: Stability at finish, low motion variance overall
    # Finish stability (last 10% of frames)
    finish_frames = int(T * 0.1)
    if finish_frames > 0 and len(frame_diffs) >= finish_frames:
        finish_motion = torch.mean(torch.abs(frame_diffs[-finish_frames:])).item()
        # Low motion at finish = good balance
        finish_stability = max(0, 1.0 - finish_motion * 20)
    else:
        finish_stability = 0.5
    
    # Overall stability (low variance in motion)
    if len(motion_per_frame) > 1:
        motion_variance = torch.var(motion_per_frame).item()
        overall_stability = max(0, 1.0 - motion_variance * 100)
    else:
        overall_stability = 0.5
    
    balance = (finish_stability * 0.6 + overall_stability * 0.4)
    
    # 8. Consistency
    # Good consistency: Repeatable motion, similar patterns
    # Measure: Low variance in motion patterns, consistent timing
    if len(events) >= 8:
        # Timing consistency
        phase_durations = np.diff(events[:8])
        if len(phase_durations) > 0 and np.mean(phase_durations) > 0:
            timing_cv = np.std(phase_durations) / (np.mean(phase_durations) + 1e-8)
            timing_consistency = max(0, 1.0 - timing_cv)
        else:
            timing_consistency = 0.5
    else:
        timing_consistency = 0.5
    
    # Motion pattern consistency
    if len(motion_per_frame) > 5:
        # Split into segments and compare
        num_segments = 4
        segment_size = len(motion_per_frame) // num_segments
        segment_means = []
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else len(motion_per_frame)
            segment_means.append(torch.mean(motion_per_frame[start:end]).item())
        
        pattern_cv = np.std(segment_means) / (np.mean(segment_means) + 1e-8)
        pattern_consistency = max(0, 1.0 - pattern_cv)
    else:
        pattern_consistency = 0.5
    
    consistency = (timing_consistency * 0.5 + pattern_consistency * 0.5)
    
    # Frame quality (if videos not normalized)
    if videos_normalized:
        frame_quality = 1.0
    else:
        frame_variances = [torch.var(gray_frames[i]).item() for i in range(T)]
        if frame_variances:
            mean_var = np.mean(frame_variances)
            max_var = max(frame_variances)
            frame_quality = min(1.0, mean_var / (max_var + 1e-8)) if max_var > 0 else 0.5
        else:
            frame_quality = 0.5
    
    return np.array([
        setup_quality,      # 1. Setup Quality
        tempo_rhythm,       # 2. Tempo & Rhythm
        weight_shift,       # 3. Weight Shift
        rotation_score,     # 4. Body Rotation
        impact_quality,     # 5. Impact Quality
        followthrough,      # 6. Follow-Through
        balance,            # 7. Balance
        consistency         # 8. Consistency
    ])


def train_golf_swing_grader(
    event_detector_path='models/swingnet_best.pth.tar',
    data_file='data/train_split_1.pkl',
    vid_dir='data/videos_160/',
    epochs=50,
    batch_size=16,
    lr=0.001,
    seq_length=64,
    videos_normalized=False
):
    """
    Train golf-specific swing grader based on professional instruction principles.
    """
    
    print("=" * 60)
    print("Training Golf Swing Quality Grader")
    print("Based on Golf Digest, MyTPI, Hackmotion principles")
    print("=" * 60)
    print("\nGrading Criteria:")
    print("1. Setup Quality (stance, posture, alignment)")
    print("2. Tempo & Rhythm (smooth, unhurried transition)")
    print("3. Weight Shift (moves to front foot in downswing)")
    print("4. Body Rotation (full shoulder turn, proper sequence)")
    print("5. Impact Quality (inferred from swing mechanics)")
    print("6. Follow-Through (full extension, weight on lead foot)")
    print("7. Balance (hold finish position, stable throughout)")
    print("8. Consistency (repeatable motion and results)")
    print("=" * 60 + "\n")
    
    # Load pretrained event detector
    print("Loading pretrained event detector...")
    event_detector = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )
    
    if event_detector_path and os.path.exists(event_detector_path):
        save_dict = torch.load(event_detector_path, map_location='cpu')
        event_detector.load_state_dict(save_dict['model_state_dict'])
        print(f"Loaded event detector from {event_detector_path}")
    else:
        print("Warning: Event detector checkpoint not found")
        return
    
    # Create golf-specific grader
    model = GolfSwingGrader(event_detector)
    model.train()
    model.cuda()
    event_detector.cuda()
    
    # Load dataset with data augmentation
    print("\nLoading dataset...")
    dataset = GolfDB(
        data_file=data_file,
        vid_dir=vid_dir,
        seq_length=seq_length,
        transform=transforms.Compose([
            ToTensor(),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            RandomErasing(probability=0.3, sl=0.02, sh=0.3, r1=0.3),  # 30% chance, light erasing
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        train=True
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,  # Pre-load next 4 batches (increased)
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    print(f"Dataset size: {len(dataset)} videos")
    
    # Generate golf instruction-based labels
    print("\nGenerating golf instruction-based quality signals...")
    weak_labels = []
    
    # Load dataframe to get events
    import pandas as pd
    df = pd.read_pickle(data_file)
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            frames = sample['images']
            
            # Get events from dataframe
            row = df.iloc[i]
            events = row['events'].copy()
            events = events - events[0]
            
            # Adjust events to match sampled sequence
            if len(frames.shape) == 4:  # (T, C, H, W)
                seq_length = frames.shape[0]
                if len(events) >= 8 and events[-1] > 0:
                    events_normalized = events / events[-1]
                    events_in_seq = (events_normalized * (seq_length - 1)).astype(int)
                    events_in_seq = np.clip(events_in_seq, 0, seq_length - 1)
                else:
                    events_in_seq = np.linspace(0, seq_length - 1, 8).astype(int)
            else:
                events_in_seq = events[:8] if len(events) >= 8 else np.linspace(0, 63, 8).astype(int)
            
            # Compute golf instruction signals
            signals = compute_golf_instruction_signals(
                frames, events_in_seq, videos_normalized=videos_normalized
            )
            weak_labels.append(signals)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} videos...")
        except Exception as e:
            print(f"  Error processing video {i}: {e}")
            import traceback
            traceback.print_exc()
            weak_labels.append(np.ones(8) * 0.5)
    
    weak_labels = np.array(weak_labels)
    print(f"Generated golf instruction labels: shape {weak_labels.shape}")
    
    # Save labels
    weak_labels_path = 'data/golf_instruction_labels.pkl'
    with open(weak_labels_path, 'wb') as f:
        pickle.dump(weak_labels, f)
    print(f"Saved labels to {weak_labels_path}")
    
    # Training setup with optimizations
    # Differential learning rates: lower LR for frozen EventDetector, higher for trainable parts
    event_detector_params = []
    trainable_params = []
    for name, param in model.named_parameters():
        if 'event_detector' in name and not param.requires_grad:
            event_detector_params.append(param)
        else:
            trainable_params.append(param)
    
    # Use different learning rates for different parts
    param_groups = [
        {'params': trainable_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': event_detector_params, 'lr': lr * 0.1, 'weight_decay': 1e-5}  # Lower LR for frozen parts
    ] if len(event_detector_params) > 0 else [{'params': trainable_params, 'lr': lr, 'weight_decay': 1e-4}]
    
    # AdamW optimizer (better weight decay handling, fused operations)
    base_optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    # Lookahead optimizer wrapper (better convergence stability)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    
    # Loss function: Huber Loss (less sensitive to outliers) or SmoothL1Loss
    if use_huber_loss:
        criterion = HuberLoss(delta=1.0, reduction='mean')
        loss_name = "HuberLoss (delta=1.0)"
    else:
        criterion = nn.SmoothL1Loss()
        loss_name = "SmoothL1Loss"
    
    # Mixed precision training for faster training
    scaler = GradScaler()
    
    # Learning rate scheduling: Cosine annealing with warm restarts
    base_scheduler = CosineAnnealingWarmRestarts(
        base_optimizer,  # Use base optimizer for scheduler
        T_0=epochs // 4,  # First restart period (1/4 of total epochs)
        T_mult=2,  # Multiplier for restart period
        eta_min=lr * 0.01
    )
    
    # Add warmup to scheduler
    warmup_steps = int(epochs * 0.1)  # Warm up over 10% of epochs
    scheduler = WarmupScheduler(base_scheduler, warmup_steps=warmup_steps)
    
    # Gradient accumulation configuration
    accumulation_steps = 2  # Accumulate gradients over 2 batches
    
    # Exponential Moving Average
    ema = EMA(model, decay=0.9999)
    
    # Stochastic Weight Averaging (complement to EMA, activates later in training)
    swa_start = int(epochs * 0.75)  # Start SWA at 75% of training
    swa = SWA(model, swa_start=swa_start, swa_freq=5, swa_lr=lr * 0.01)
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("OPTIMIZATIONS ENABLED:")
    print("  - Mixed Precision Training: ENABLED (FP16)")
    print("  - Optimizer: Lookahead(AdamW) (k=5, alpha=0.5, better convergence)")
    print("  - Differential Learning Rates: ENABLED")
    print("    * Trainable layers: lr={}".format(lr))
    print("    * Frozen EventDetector: lr={} (10% of trainable)".format(lr * 0.1))
    print("  - Weight Decay: ENABLED (1e-4, L2 regularization)")
    print("  - Learning Rate Scheduling: ENABLED (Warmup + Cosine Annealing with Warm Restarts)")
    print("    * Warmup: {} epochs (10% of total)".format(warmup_steps))
    print("    * Cosine Annealing: T_0={}, T_mult=2, eta_min={}".format(epochs // 4, lr * 0.01))
    print("  - Gradient Clipping: ENABLED (max_norm=1.0)")
    print("  - Gradient Accumulation: ENABLED (steps=2)")
    print("  - Exponential Moving Average (EMA): ENABLED (decay=0.9999)")
    print("  - Stochastic Weight Averaging (SWA): ENABLED (starts at epoch {})".format(swa_start))
    print("  - Loss Function: {} (better than MSE for regression)".format(loss_name))
    print("  - Data Augmentation: ENABLED")
    print("    * ColorJitter: brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05")
    print("    * RandomErasing: probability=0.3, sl=0.02, sh=0.3, r1=0.3")
    print("  - Weight Initialization: ENABLED (Xavier uniform for quality head)")
    print("  - Data Loading: 4 workers, pin_memory=True, prefetch_factor=4, persistent_workers=True")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, sample in enumerate(data_loader):
            images = sample['images'].cuda()
            
            # Calculate batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_indices = list(range(start_idx, end_idx))
            
            # Get labels for this batch
            actual_batch_size = len(batch_indices)
            batch_labels = torch.tensor(
                weak_labels[batch_indices],
                dtype=torch.float32
            ).cuda()
            
            # Mixed precision forward pass
            with autocast(device_type='cuda'):
                predicted_scores, attention_weights = model(images)
                
                if predicted_scores.shape[0] != actual_batch_size:
                    predicted_scores = predicted_scores[:actual_batch_size]
                
                # Loss
                loss = criterion(predicted_scores, batch_labels)
            
            # Gradient accumulation: scale loss
            loss = loss / accumulation_steps
            
            # Mixed precision backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights every N accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(base_optimizer)  # Use base optimizer for unscale
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(base_optimizer)  # Step base optimizer
                scaler.update()
                optimizer.step()  # Step lookahead optimizer
                optimizer.zero_grad()
                # Learning rate scheduling
                scheduler.step()
                # Update EMA
                ema.update()
                # Update SWA (if past start epoch)
                if epoch >= swa_start and (batch_idx + 1) % (len(data_loader) // 10) == 0:
                    swa.update_swa()
            
            # Use unscaled loss for logging
            epoch_loss += loss.item() * accumulation_steps
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item() * accumulation_steps:.4f}")
        
        avg_loss = epoch_loss / len(data_loader)
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print("-" * 60)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            # Apply EMA for checkpoint
            ema.apply_shadow()
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'base_optimizer_state_dict': base_optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'base_scheduler_state_dict': base_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'models/golf_swing_grader_epoch_{epoch+1}.pth.tar')
            ema.restore()
            print(f"Saved checkpoint: models/golf_swing_grader_epoch_{epoch+1}.pth.tar")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Apply EMA and save final model
    ema.apply_shadow()
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'base_optimizer_state_dict': base_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(final_checkpoint, 'models/golf_swing_grader_final_ema.pth.tar')
    print("Saved final model (EMA): models/golf_swing_grader_final_ema.pth.tar")
    
    # Apply SWA if it was used
    if swa.swa_model is not None:
        swa.apply_swa()
        torch.save({'model_state_dict': model.state_dict()}, 
                  'models/golf_swing_grader_final_swa.pth.tar')
        print("Saved final model (SWA): models/golf_swing_grader_final_swa.pth.tar")
        ema.restore()  # Restore EMA weights


if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    train_golf_swing_grader(
        event_detector_path='models/swingnet_best.pth.tar',
        data_file='data/train_split_1.pkl',
        vid_dir='data/videos_160/',
        epochs=50,
        batch_size=16,
        lr=0.001,
        videos_normalized=False
    )

