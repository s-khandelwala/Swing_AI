"""
Analyze Golf Swing Video
Crops video using pose-based detection, runs through event detector, and outputs annotated video.
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import os.path as osp
import subprocess
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import motion detector (now in root)
from motion_swing_detector import SwingBoundaryDetector, PoseBasedSwingDetector

# Import model components
from model import EventDetector
from MobileNetV2 import MobileNetV2

# Golf swing event names (8 events + background)
EVENT_NAMES = [
    'Address',
    'Takeaway', 
    'Backswing',
    'Top',
    'Downswing',
    'Impact',
    'Follow-through',
    'Finish',
    'Background'
]


def get_video_rotation(video_path):
    """
    Detect video rotation from metadata using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Rotation in degrees (0, 90, 180, or 270)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Check for rotation tag first (most reliable)
                rotation = stream.get('tags', {}).get('rotate')
                if rotation:
                    rot_int = int(rotation)
                    rot_int = rot_int % 360
                    if rot_int == 90:
                        return 90
                    elif rot_int == 180:
                        return 180
                    elif rot_int == 270:
                        return 270
                
                # Check displaymatrix side data
                side_data_list = stream.get('side_data_list', [])
                for side_data in side_data_list:
                    if side_data.get('side_data_type') == 'Display Matrix':
                        matrix_str = side_data.get('displaymatrix', '')
                        if '-65536' in matrix_str or '65536' in matrix_str:
                            return 90
    except Exception:
        pass  # If ffprobe fails, assume no rotation
    
    return 0


def rotate_frame(frame, rotation_degrees):
    """
    Rotate a frame by the specified degrees.
    
    Args:
        frame: Input frame (BGR format)
        rotation_degrees: Rotation in degrees (0, 90, 180, or 270)
        
    Returns:
        Rotated frame
    """
    if rotation_degrees == 0:
        return frame
    elif rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame  # Unknown rotation, return as-is


def detect_slowmo_by_peak_sharpness(video_path, swing_start_frame=0, swing_end_frame=None):
    """
    Detect slow-motion by measuring acceleration peak sharpness.
    Uses relative timing (rise width) instead of absolute timing.
    
    Args:
        video_path: Path to video file
        swing_start_frame: Start frame of swing (0-indexed)
        swing_end_frame: End frame of swing (None = use all frames)
        
    Returns:
        (is_slowmo, confidence, speed_factor_estimate)
        - is_slowmo: True if slow-motion detected
        - confidence: 0.0 to 1.0 (how confident we are)
        - speed_factor: Estimated speed multiplier (1.0 = normal, 2.0 = 2x slow-mo)
    """
    from motion_swing_detector import PoseBasedSwingDetector
    
    print(f"\n{'='*60}")
    print("Detecting slow-motion using peak sharpness analysis...")
    print(f"{'='*60}")
    
    detector = PoseBasedSwingDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False, 0.0, 1.0
    
    if swing_end_frame is None:
        swing_end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    
    # Extract motion scores for the swing segment
    motion_scores = []
    keypoints_prev = None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, swing_start_frame)
    frame_count = swing_start_frame
    
    print(f"Analyzing motion from frame {swing_start_frame} to {swing_end_frame}...")
    
    while frame_count <= swing_end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints = detector._extract_keypoints(frame)
        
        if keypoints is not None:
            if keypoints_prev is not None:
                # Compute motion score using the same method as swing detection
                motion_score = detector._compute_motion_score(keypoints_prev, keypoints)
                motion_scores.append(motion_score)
            keypoints_prev = keypoints
        else:
            # If pose not detected, use previous motion or 0
            if len(motion_scores) > 0:
                motion_scores.append(motion_scores[-1])
            else:
                motion_scores.append(0.0)
        
        frame_count += 1
    
    cap.release()
    
    if len(motion_scores) < 10:
        print("⚠️  Not enough motion data for slow-mo detection")
        return False, 0.0, 1.0
    
    motion_scores = np.array(motion_scores)
    
    # Smooth motion scores to reduce noise
    try:
        from scipy.ndimage import uniform_filter1d
        motion_scores = uniform_filter1d(motion_scores, size=3)
    except (ImportError, AttributeError):
        # Fallback: simple moving average if scipy not available
        kernel = np.ones(3) / 3
        motion_scores = np.convolve(motion_scores, kernel, mode='same')
    
    # Find peak
    peak_idx = np.argmax(motion_scores)
    peak_value = motion_scores[peak_idx]
    motion_mean = np.mean(motion_scores)
    motion_std = np.std(motion_scores)
    
    print(f"   Motion statistics: peak={peak_value:.4f}, mean={motion_mean:.4f}, std={motion_std:.4f}")
    
    if peak_value < 1e-6:
        print("⚠️  No significant motion detected")
        return False, 0.0, 1.0
    
    # Low peak value relative to typical swings might indicate slow-mo
    # Typical real-time swings have peak motion > 0.05, slow-mo often < 0.03
    if peak_value < 0.03:
        print(f"   ⚠️  Very low peak motion ({peak_value:.4f}) - likely slow-motion")
    
    # Measure rise width: frames to go from 20% to 80% of peak
    # This measures how "sharp" the acceleration spike is
    # Handle edge case: if peak is at start, look at the rise after peak
    if peak_idx == 0:
        # Peak at start - measure fall instead (from peak to 20% after peak)
        fall_start_indices = np.where(motion_scores[peak_idx:] <= 0.8 * peak_value)[0]
        fall_end_indices = np.where(motion_scores[peak_idx:] <= 0.2 * peak_value)[0]
        
        if len(fall_start_indices) > 0 and len(fall_end_indices) > 0:
            fall_start = fall_start_indices[0]
            fall_end = fall_end_indices[-1]
            fall_width = fall_end - fall_start
            total_swing_frames = len(motion_scores)
            fall_ratio = fall_width / total_swing_frames if total_swing_frames > 0 else 0
            
            # Use fall ratio as indicator (wide fall = slow-mo)
            is_slowmo = fall_ratio > 0.20
            confidence = min(1.0, abs(fall_ratio - 0.15) / 0.15)
            speed_factor = max(1.0, min(4.0, fall_ratio / 0.10))
            
            print(f"   Peak at frame {peak_idx} (value: {peak_value:.3f}) - at start")
            print(f"   Fall width: {fall_width} frames ({fall_ratio:.3f} of total)")
            print(f"   Detection: {'SLOW-MOTION' if is_slowmo else 'REAL-TIME'}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Estimated speed factor: {speed_factor:.2f}x")
            
            return is_slowmo, confidence, speed_factor
    
    # Normal case: measure rise before peak
    rise_start_indices = np.where(motion_scores[:peak_idx] >= 0.2 * peak_value)[0]
    rise_end_indices = np.where(motion_scores[:peak_idx] >= 0.8 * peak_value)[0]
    
    if len(rise_start_indices) > 0 and len(rise_end_indices) > 0:
        rise_start = rise_start_indices[0]
        rise_end = rise_end_indices[-1]
        rise_width = rise_end - rise_start
        
        total_swing_frames = len(motion_scores)
        rise_ratio = rise_width / total_swing_frames if total_swing_frames > 0 else 0
        
        # Threshold: real swings have sharp peaks (ratio < 0.15)
        # Slow-mo has stretched peaks (ratio > 0.20)
        # This is relative timing, not absolute!
        is_slowmo = rise_ratio > 0.20
        
        # Confidence: how far from the threshold
        # Real swings typically have ratio ~0.10, slow-mo ~0.30+
        if is_slowmo:
            confidence = min(1.0, (rise_ratio - 0.20) / 0.20)  # 0.20 -> 0.0, 0.40 -> 1.0
        else:
            confidence = min(1.0, (0.20 - rise_ratio) / 0.20)  # 0.20 -> 0.0, 0.00 -> 1.0
        
        # Estimate speed factor: wider peak = more slow-mo
        # Normalize to expected real-time ratio (~0.10)
        speed_factor = max(1.0, min(4.0, rise_ratio / 0.10))
        
        print(f"   Peak at frame {peak_idx} (value: {peak_value:.3f})")
        print(f"   Rise width: {rise_width} frames ({rise_ratio:.3f} of total)")
        print(f"   Detection: {'SLOW-MOTION' if is_slowmo else 'REAL-TIME'}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Estimated speed factor: {speed_factor:.2f}x")
        
        return is_slowmo, confidence, speed_factor
    else:
        # Fallback: use overall motion distribution
        # If motion is very spread out, likely slow-mo
        motion_std = np.std(motion_scores)
        motion_mean = np.mean(motion_scores)
        coefficient_of_variation = motion_std / (motion_mean + 1e-6)
        
        # Low variation = smooth/stretched motion = slow-mo
        # High variation = sharp spikes = real-time
        is_slowmo = coefficient_of_variation < 0.5
        
        # Fallback: use multiple indicators
        # 1. Peak width (how many frames above 50% of peak)
        frames_above_half = np.sum(motion_scores >= 0.5 * peak_value)
        peak_width_ratio = frames_above_half / len(motion_scores)
        
        # 2. Low peak value (slow-mo has lower per-frame motion)
        low_peak_indicator = 1.0 if peak_value < 0.03 else (1.0 - (peak_value - 0.03) / 0.05)
        low_peak_indicator = max(0.0, min(1.0, low_peak_indicator))
        
        # 3. Motion smoothness (slow-mo has smoother, less variable motion)
        # Coefficient of variation: low = smooth = slow-mo
        cv = motion_std / (motion_mean + 1e-6)
        smoothness_indicator = 1.0 if cv < 0.8 else max(0.0, 1.0 - (cv - 0.8) / 0.5)
        
        # 4. Overall motion level (slow-mo has consistently low motion)
        low_motion_indicator = 1.0 if motion_mean < 0.015 else max(0.0, 1.0 - (motion_mean - 0.015) / 0.02)
        
        # Combine indicators (weighted average)
        slowmo_score = (
            0.3 * min(1.0, peak_width_ratio / 0.15) +  # Peak width (lower threshold: 0.15)
            0.3 * low_peak_indicator +                   # Low peak value
            0.2 * smoothness_indicator +                 # Motion smoothness
            0.2 * low_motion_indicator                   # Overall low motion
        )
        
        is_slowmo = slowmo_score > 0.5
        confidence = abs(slowmo_score - 0.5) * 2  # Convert to 0-1 confidence
        
        # Estimate speed factor based on multiple factors
        if is_slowmo:
            # Higher slowmo_score = more slow-mo = higher speed factor
            speed_factor = 1.0 + (slowmo_score - 0.5) * 6.0  # 0.5 -> 1.0x, 1.0 -> 4.0x
            speed_factor = max(1.5, min(4.0, speed_factor))
        else:
            speed_factor = 1.0
        
        print(f"   Peak at frame {peak_idx} (value: {peak_value:.4f})")
        print(f"   Using fallback method with multiple indicators:")
        print(f"     - Peak width ratio: {peak_width_ratio:.3f}")
        print(f"     - Low peak indicator: {low_peak_indicator:.3f}")
        print(f"     - Smoothness indicator: {smoothness_indicator:.3f}")
        print(f"     - Low motion indicator: {low_motion_indicator:.3f}")
        print(f"     - Combined slowmo score: {slowmo_score:.3f}")
        print(f"   Detection: {'SLOW-MOTION' if is_slowmo else 'REAL-TIME'}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Estimated speed factor: {speed_factor:.2f}x")
        
        return is_slowmo, confidence, speed_factor


def speed_up_slow_motion(video_path, output_path, speed_factor=2.0, slow_regions=None):
    """
    Speed up slow-motion regions in a video using accurate temporal sampling.
    Uses accumulator method for precise frame selection, even with non-integer speed factors.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        speed_factor: Speed multiplier (2.0 = 2x speed, 2.5 = 2.5x speed)
        slow_regions: List of (start_frame, end_frame) tuples. If None, speed up entire video.
        
    Returns:
        (fps, width, height) of output video
    """
    print(f"\n{'='*60}")
    print(f"Speeding up slow-motion by {speed_factor}x...")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a mask for slow-motion frames if regions specified
    slow_mask = None
    if slow_regions:
        slow_mask = np.zeros(total_frames, dtype=bool)
        for start, end in slow_regions:
            slow_mask[start:end+1] = True
        print(f"   Speeding up {np.sum(slow_mask)} frames in {len(slow_regions)} region(s)")
    else:
        print(f"   Speeding up entire video ({total_frames} frames)")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")
    
    # Accurate temporal sampling using accumulator method
    # This ensures exact speed_factor, even for non-integer values
    # For speed_factor = 2.5: we keep 1 frame every 2.5 frames
    # Accumulator tracks fractional progress: += 1/speed_factor each frame
    # When accumulator >= 1.0, write frame and subtract 1.0
    frame_count = 0
    frames_written = 0
    accumulator = 0.0  # Accumulator for fractional frame selection
    
    print(f"   Using temporal sampling (step = 1/{speed_factor:.3f} = {1.0/speed_factor:.3f})")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        is_slow = slow_mask[frame_count] if slow_mask is not None else True
        
        if is_slow:
            # Slow-motion region: use accumulator for accurate sampling
            accumulator += 1.0 / speed_factor
            
            if accumulator >= 1.0:
                # Write this frame
                out.write(frame)
                frames_written += 1
                accumulator -= 1.0
                # Keep remainder for next iteration (ensures exact speed_factor)
        else:
            # Normal speed: write all frames
            out.write(frame)
            frames_written += 1
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Calculate actual speed achieved
    actual_speed = total_frames / frames_written if frames_written > 0 else 1.0
    speed_accuracy = (actual_speed / speed_factor) * 100 if speed_factor > 0 else 0
    
    print(f"✅ Speeded-up video saved: {output_path}")
    print(f"   Original: {total_frames} frames")
    print(f"   Output: {frames_written} frames")
    print(f"   Target speed: {speed_factor:.3f}x")
    print(f"   Actual speed: {actual_speed:.3f}x ({speed_accuracy:.1f}% of target)")
    
    return fps, width, height


def load_event_detector(model_path='models/swingnet_2000.pth.tar', device='cuda'):
    """Load the trained event detector model."""
    print(f"Loading event detector from {model_path}...")
    
    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    print("✅ Event detector loaded successfully")
    return model


def crop_video(video_path, start_frame, end_frame, output_path):
    """
    Crop video to specified frame range, with automatic rotation correction.
    
    Args:
        video_path: Input video path
        start_frame: Start frame (0-indexed)
        end_frame: End frame (0-indexed, inclusive)
        output_path: Output video path
        
    Returns:
        (fps, width, height) of cropped video
    """
    print(f"\n{'='*60}")
    print("Cropping video...")
    print(f"{'='*60}")
    
    # Detect video rotation
    rotation = get_video_rotation(video_path)
    if rotation != 0:
        print(f"Detected video rotation: {rotation} degrees (will be corrected)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {total_frames} frames ({total_frames/fps:.2f}s)")
    print(f"Cropping to frames {start_frame}-{end_frame} ({end_frame-start_frame+1} frames)")
    
    # Validate frame range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    
    # Determine output dimensions after rotation
    if rotation in [90, 270]:
        output_width, output_height = height, width
    else:
        output_width, output_height = width, height
    
    # Set up video writer with corrected dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    frames_written = 0
    
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation if needed
        if rotation != 0:
            frame = rotate_frame(frame, rotation)
        
        out.write(frame)
        frames_written += 1
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"✅ Cropped video saved: {output_path}")
    print(f"   Frames: {frames_written} ({frames_written/fps:.2f}s)")
    if rotation != 0:
        print(f"   Rotation corrected: {rotation}° → 0°")
    
    return fps, output_width, output_height


def preprocess_frame(frame, size=(160, 160)):
    """
    Preprocess frame for event detector.
    
    Args:
        frame: BGR frame from OpenCV
        size: Target size (width, height)
        
    Returns:
        Preprocessed tensor (C, H, W)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    frame_resized = cv2.resize(frame_rgb, size)  # (H, W, C)
    
    # Convert to tensor: (H, W, C) -> (C, H, W) and normalize to [0, 1]
    frame_tensor = torch.from_numpy(frame_resized).float()
    frame_tensor = frame_tensor.permute(2, 0, 1)  # (C, H, W)
    frame_tensor = frame_tensor / 255.0  # Normalize to [0, 1]
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frame_tensor = (frame_tensor - mean) / std
    
    return frame_tensor


def find_prominent_peaks(prob_curve, min_prominence_ratio=0.2, min_height_ratio=0.05, window_size=10):
    """
    Find prominent peaks in a probability curve using data-driven thresholds.
    
    Args:
        prob_curve: Probability values across frames
        min_prominence_ratio: Peak must be this ratio above local baseline (0.2 = 20% above)
        min_height_ratio: Peak must be at least this ratio of max value (0.05 = 5% of max)
        window_size: Window size for local baseline calculation
    
    Returns:
        List of (peak_idx, prominence, height) tuples, sorted by prominence (most prominent first)
    """
    peaks = []
    max_prob = np.max(prob_curve)
    baseline = np.percentile(prob_curve, 25)  # 25th percentile as global baseline
    
    if max_prob < 1e-6:
        return peaks  # No signal
    
    # Find local maxima
    for i in range(1, len(prob_curve) - 1):
        if prob_curve[i] > prob_curve[i-1] and prob_curve[i] > prob_curve[i+1]:
            # Local maximum found
            peak_height = prob_curve[i]
            
            # Calculate prominence: how much above local baseline
            # Look left and right for local minima within window
            left_start = max(0, i - window_size)
            right_end = min(len(prob_curve), i + window_size + 1)
            
            left_min = np.min(prob_curve[left_start:i]) if i > left_start else peak_height
            right_min = np.min(prob_curve[i:right_end]) if right_end > i else peak_height
            local_baseline = min(left_min, right_min)
            
            prominence = peak_height - local_baseline
            
            # Relative prominence (as ratio of peak height)
            if peak_height > 0:
                rel_prominence = prominence / peak_height
            else:
                rel_prominence = 0.0
            
            # Relative height (as ratio of max probability)
            rel_height = peak_height / (max_prob + 1e-6)
            
            # Only keep prominent peaks
            if rel_prominence >= min_prominence_ratio and rel_height >= min_height_ratio:
                peaks.append((i, rel_prominence, rel_height, peak_height))
    
    # Sort by prominence (most prominent first)
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks


def predict_events(model, video_path, device='cuda', seq_length=64):
    """
    Predict swing events from cropped video using adaptive peak detection.
    
    Args:
        model: EventDetector model
        video_path: Path to cropped video
        device: Device to run model on
        seq_length: Sequence length for processing
        
    Returns:
        List of predicted event frames (8 events)
    """
    print(f"\n{'='*60}")
    print("Predicting swing events...")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    total_frames = len(frames)
    print(f"Processing {total_frames} frames...")
    
    # Preprocess all frames
    print("Preprocessing frames...")
    processed_frames = []
    for i, frame in enumerate(frames):
        if (i + 1) % 50 == 0:
            print(f"  Preprocessed {i+1}/{total_frames} frames...")
        tensor = preprocess_frame(frame)
        processed_frames.append(tensor)
    
    # Stack into tensor: (T, C, H, W)
    frames_tensor = torch.stack(processed_frames)  # (T, C, H, W)
    frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W) - add batch dimension
    
    # Process in batches
    print("Running event detection...")
    all_probs = []
    
    batch = 0
    while batch * seq_length < total_frames:
        start_idx = batch * seq_length
        end_idx = min((batch + 1) * seq_length, total_frames)
        
        batch_frames = frames_tensor[:, start_idx:end_idx, :, :, :].to(device)
        
        with torch.no_grad():
            logits = model(batch_frames)  # (batch*timesteps, 9)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (timesteps, 9)
        
        if batch == 0:
            all_probs = probs
        else:
            all_probs = np.append(all_probs, probs, axis=0)
        
        batch += 1
        if batch % 10 == 0:
            print(f"  Processed {end_idx}/{total_frames} frames...")
    
    print(f"✅ Event detection complete")
    print(f"   Probability matrix shape: {all_probs.shape} (frames x classes)")
    print(f"   Average max probability per frame: {np.mean(np.max(all_probs, axis=1)):.3f}")
    print(f"   Average background probability: {np.mean(all_probs[:, 8]):.3f}")
    
    # Get background probabilities for signal-to-noise calculation
    background_probs = all_probs[:, 8]
    
    # DIAGNOSTICS: Show probability statistics for each event
    print(f"\n   📊 Event probability diagnostics:")
    for event_idx in range(8):
        event_probs = all_probs[:, event_idx]
        max_prob = np.max(event_probs)
        mean_prob = np.mean(event_probs)
        std_prob = np.std(event_probs)
        median_prob = np.median(event_probs)
        
        # Find top 3 frames with highest probability
        top_3_indices = np.argsort(event_probs)[-3:][::-1]
        top_3_probs = event_probs[top_3_indices]
        
        # Calculate signal-to-noise for top frames
        top_3_snr = event_probs[top_3_indices] / (background_probs[top_3_indices] + 1e-6)
        
        print(f"     {EVENT_NAMES[event_idx]:15s}:")
        print(f"       Max: {max_prob:.4f}, Mean: {mean_prob:.4f}, Median: {median_prob:.4f}, Std: {std_prob:.4f}")
        print(f"       Top 3 frames: {top_3_indices} (probs: {top_3_probs}, SNR: {top_3_snr})")
    
    # Extract event frames using global optimization with temporal ordering
    # This finds the best set of frames that maximize total score while maintaining order
    print(f"\n   🔍 Finding globally optimal event frames...")
    
    # Calculate signal-to-noise ratio for all events
    signal_to_noise = np.zeros_like(all_probs)
    for event_idx in range(8):
        event_probs = all_probs[:, event_idx]
        signal_to_noise[:, event_idx] = event_probs / (background_probs + 1e-6)
    
    # Apply light smoothing to SNR
    window_size = 5
    kernel = np.ones(window_size) / window_size
    for event_idx in range(8):
        signal_to_noise[:, event_idx] = np.convolve(signal_to_noise[:, event_idx], kernel, mode='same')
    
    # For each event, find candidate frames (local peaks with high probability/SNR)
    # We'll use these candidates in the optimization
    candidate_frames = []
    candidate_probs = []
    candidate_snr = []
    
    for event_idx in range(8):
        event_probs = all_probs[:, event_idx]
        event_snr = signal_to_noise[:, event_idx]
        
        # Find local peaks (frames that are higher than neighbors)
        candidates = []
        for frame in range(1, total_frames - 1):
            if event_probs[frame] > event_probs[frame - 1] and event_probs[frame] > event_probs[frame + 1]:
                # Local peak found - use probability as primary score
                candidates.append((frame, event_probs[frame], event_snr[frame]))
        
        # Also include the global maximum
        max_frame = np.argmax(event_probs)
        if (max_frame, event_probs[max_frame], event_snr[max_frame]) not in candidates:
            candidates.append((max_frame, event_probs[max_frame], event_snr[max_frame]))
        
        # Sort by probability (descending) - we want high confidence detections
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top candidates (at least top 5, but include any with prob > 0.01)
        top_candidates = []
        for c in candidates:
            if c[1] > 0.01 or len(top_candidates) < 5:
                top_candidates.append(c)
            if len(top_candidates) >= 10:
                break
        
        candidate_frames.append([c[0] for c in top_candidates])
        candidate_probs.append([c[1] for c in top_candidates])
        candidate_snr.append([c[2] for c in top_candidates])
    
    # New approach: For each frame, find the event with highest probability
    # Then enforce temporal ordering constraints
    def find_optimal_sequence():
        """Find best event frames using candidate-based optimization with before/after offset logic."""
        # For efficiency, limit search space to top candidates
        max_candidates_per_event = 5
        
        # Build search space for all events
        search_space = []
        for event_idx in range(8):
            candidates = candidate_frames[event_idx][:max_candidates_per_event]
            if len(candidates) == 0:
                # Fallback: use frame with max probability
                candidates = [np.argmax(all_probs[:, event_idx])]
            search_space.append(candidates)
        
        # Place all events EXCEPT Top first, then place Top last
        # Event order: 0=Address, 1=Takeaway, 2=Backswing, 3=Top, 4=Downswing, 5=Impact, 6=Follow-through, 7=Finish
        best_frames = [None] * 8
        
        # Place events in this order: Address, Takeaway, Backswing, Downswing, Impact, Follow-through, Finish (skip Top)
        event_order = [0, 1, 2, 4, 5, 6, 7]  # Skip Top (3) for now
        
        for event_idx in event_order:
            prev_frame = -1
            # Find the latest placed event before this one
            for prev_idx in range(event_idx):
                if best_frames[prev_idx] is not None:
                    prev_frame = max(prev_frame, best_frames[prev_idx])
            
            # Find best candidate for this event that's after previous
            best_candidate = None
            best_score = -np.inf
            
            for candidate_frame in search_space[event_idx]:
                if candidate_frame > prev_frame:  # Must maintain temporal order
                    # Score is probability (primary) - prioritize high confidence
                    prob_score = all_probs[candidate_frame, event_idx]
                    snr_score = signal_to_noise[candidate_frame, event_idx]
                    combined_score = prob_score * 2.0 + snr_score * 0.5
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate_frame
            
            # If no valid candidate found, use the first frame after previous
            if best_candidate is None:
                # Search for any frame after previous with non-zero probability
                for frame in range(prev_frame + 1, total_frames):
                    if all_probs[frame, event_idx] > 0:
                        best_candidate = frame
                        break
                
                # Last resort: just use next frame
                if best_candidate is None:
                    best_candidate = min(prev_frame + 1, total_frames - 1)
            
            best_frames[event_idx] = best_candidate
        
        # Now place Top (event_idx 3) - find best position between Backswing (2) and Downswing (4)
        backswing_frame = best_frames[2] if best_frames[2] is not None else 0
        downswing_frame = best_frames[4] if best_frames[4] is not None else total_frames - 1
        
        # Find best Top candidate between Backswing and Downswing
        best_top_candidate = None
        best_top_score = -np.inf
        
        for candidate_frame in search_space[3]:  # Top candidates
            if backswing_frame < candidate_frame < downswing_frame:
                # This candidate fits between Backswing and Downswing
                prob_score = all_probs[candidate_frame, 3]
                snr_score = signal_to_noise[candidate_frame, 3]
                combined_score = prob_score * 2.0 + snr_score * 0.5
                
                if combined_score > best_top_score:
                    best_top_score = combined_score
                    best_top_candidate = candidate_frame
        
        # If no candidate found in the gap, use the frame with highest probability in that range
        if best_top_candidate is None:
            search_start = backswing_frame + 1
            search_end = downswing_frame
            if search_end > search_start:
                top_probs_in_range = all_probs[search_start:search_end, 3]
                if np.max(top_probs_in_range) > 0:
                    best_top_candidate = search_start + np.argmax(top_probs_in_range)
                else:
                    # Use midpoint if no probability in range
                    best_top_candidate = (backswing_frame + downswing_frame) // 2
            else:
                # Gap is too small, use frame right after Backswing
                best_top_candidate = min(backswing_frame + 1, total_frames - 1)
        
        best_frames[3] = best_top_candidate
        
        # Apply before/after offset logic based on event type
        # For pre-Top events (0-2): use frame BEFORE the detected frame
        # For Top (3): use the detected frame itself
        # For post-Top events (4-7): use frame AFTER the detected frame
        # BUT: We want to stay as close to the peak as possible, so prefer the peak frame if it has higher probability
        event_frames = []
        for event_idx, detected_frame in enumerate(best_frames):
            if event_idx < 3:
                # Pre-Top: use frame BEFORE (everything before is that phase)
                # But prefer the peak frame if it has higher probability
                offset_frame = max(0, detected_frame - 1)
                prob_at_offset = all_probs[offset_frame, event_idx]
                prob_at_detected = all_probs[detected_frame, event_idx]
                # Use the frame with higher probability
                if prob_at_detected > prob_at_offset:
                    event_frames.append(detected_frame)  # Keep the peak
                else:
                    event_frames.append(offset_frame)
            elif event_idx == 3:
                # Top: use the frame itself (both before and after)
                event_frames.append(detected_frame)
            else:
                # Post-Top: use frame AFTER (everything after is that phase)
                # But prefer the peak frame if it has higher probability
                offset_frame = min(total_frames - 1, detected_frame + 1)
                prob_at_detected = all_probs[detected_frame, event_idx]
                prob_at_offset = all_probs[offset_frame, event_idx]
                # Use the frame with higher probability
                if prob_at_detected > prob_at_offset:
                    event_frames.append(detected_frame)  # Keep the peak
                else:
                    event_frames.append(offset_frame)
        
        # Ensure strict ordering (each event after previous)
        for event_idx in range(1, 8):
            if event_frames[event_idx] <= event_frames[event_idx - 1]:
                event_frames[event_idx] = event_frames[event_idx - 1] + 1
                if event_frames[event_idx] >= total_frames:
                    event_frames[event_idx] = total_frames - 1
        
        return event_frames
    
    # Try the optimization
    event_frames = find_optimal_sequence()
    
    # Get confidences for selected frames
    event_confidences = []
    for event_idx, frame in enumerate(event_frames):
        confidence = all_probs[frame, event_idx]
        event_confidences.append(float(confidence))
        
        # Calculate metrics
        snr_value = signal_to_noise[frame, event_idx]
        bg_prob_at_frame = background_probs[frame]
        relative_confidence = confidence / (bg_prob_at_frame + 1e-6)
        
        # Show detection info
        print(f"  {EVENT_NAMES[event_idx]:15s}: Frame {frame:4d} "
              f"(abs: {confidence:.3f}, SNR: {snr_value:.3f}, rel: {relative_confidence:.2f}x vs bg)")
        
        # Warn about very low confidence or SNR
        if confidence < 0.01:
            print(f"     ⚠️  Extremely low confidence ({confidence:.3f}) - model may not recognize this event")
        elif confidence < 0.05:
            print(f"     ⚠️  Very low confidence ({confidence:.3f}) - detection may be unreliable")
        elif snr_value < 0.1:
            print(f"     ⚠️  Low signal-to-noise ({snr_value:.3f}) - event barely distinguishable from background")
    
    return event_frames, event_confidences, all_probs


def plot_event_probabilities(all_probs, event_frames, event_confidences, output_path):
    """
    Create a graph showing probability curves for each event over time.
    
    Args:
        all_probs: Probability matrix (frames x 9) - 8 events + background
        event_frames: List of detected event frame numbers (8 events)
        event_confidences: List of confidence scores (8 events)
        output_path: Path to save the graph (PNG file)
    """
    print(f"\n   📈 Creating probability graph...")
    
    total_frames = all_probs.shape[0]
    frames = np.arange(total_frames)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Event Detection Probabilities Over Time', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each event (0-7) + background (8)
    for event_idx in range(9):
        ax = axes_flat[event_idx]
        event_probs = all_probs[:, event_idx]
        event_name = EVENT_NAMES[event_idx] if event_idx < 8 else 'Background'
        
        # Plot probability curve
        ax.plot(frames, event_probs, linewidth=1.5, alpha=0.7, label=event_name)
        
        # Mark detected event frame (if this is an event, not background)
        if event_idx < 8:
            detected_frame = event_frames[event_idx]
            detected_conf = event_confidences[event_idx]
            ax.scatter([detected_frame], [detected_conf], 
                      color='red', s=100, zorder=5, marker='x', linewidths=3,
                      label=f'Detected (frame {detected_frame})')
            
            # Add text annotation
            ax.annotate(f'Frame {detected_frame}\n({detected_conf:.3f})',
                       xy=(detected_frame, detected_conf),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=8, ha='left')
        
        # Styling
        ax.set_title(event_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim([0, max(0.1, np.max(event_probs) * 1.1)])  # Auto-scale, but at least 0-0.1
        
        # Add statistics text
        max_prob = np.max(event_probs)
        mean_prob = np.mean(event_probs)
        max_frame = np.argmax(event_probs)
        stats_text = f'Max: {max_prob:.4f} @ frame {max_frame}\nMean: {mean_prob:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove the 9th subplot (we only have 9 events: 0-7 + background)
    fig.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Probability graph saved: {output_path}")


def annotate_video(video_path, event_frames, event_confidences, output_path):
    """
    Create annotated video showing swing phases.
    
    Args:
        video_path: Input video path
        event_frames: List of 8 event frame numbers
        event_confidences: List of 8 confidence scores
        output_path: Output video path
    """
    print(f"\n{'='*60}")
    print("Creating annotated video...")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Annotating {total_frames} frames...")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color scheme for each phase
    phase_colors = [
        (0, 255, 0),      # Address - Green
        (255, 200, 0),    # Takeaway - Orange
        (255, 150, 0),    # Backswing - Dark Orange
        (255, 100, 0),    # Top - Red-Orange
        (255, 0, 0),      # Downswing - Red
        (200, 0, 255),    # Impact - Purple
        (150, 0, 255),    # Follow-through - Blue-Purple
        (0, 150, 255),    # Finish - Blue
    ]
    
    def get_current_phase(frame_num):
        """Get current phase index based on event frames.
        
        For pre-Top events (0-3): The phase includes the event frame and everything up to it (from previous event)
        For post-Top events (4-7): The phase includes the event frame and everything after it (until next event)
        
        Example:
        - Address at frame 15: frames 0-15 are Address (inclusive)
        - Impact at frame 50: frames 50-59 are Impact (inclusive, until Follow-through at 60)
        """
        # Pre-Top events (0-3: Address, Takeaway, Backswing, Top)
        # These phases include the event frame and everything up to it
        for i in range(4):  # 0-3
            prev_frame = event_frames[i-1] if i > 0 else -1
            if prev_frame < frame_num <= event_frames[i]:
                return i
        
        # Post-Top events (4-7: Downswing, Impact, Follow-through, Finish)
        # These phases include the event frame and everything after it
        for i in range(4, 8):  # 4-7
            next_frame = event_frames[i+1] if i < 7 else total_frames
            if event_frames[i] <= frame_num < next_frame:
                return i
        
        # Fallback: should not reach here, but return Finish if we do
        return 7
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get current phase
        phase_idx = get_current_phase(frame_num)
        phase_name = EVENT_NAMES[phase_idx]
        phase_color = phase_colors[phase_idx]
        
        # Draw phase name (large, top-left)
        cv2.putText(frame, phase_name, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, phase_color, 3)
        
        # Draw phase bar at top
        bar_height = 10
        bar_y = 10
        bar_width = int((frame_num / total_frames) * width) if total_frames > 0 else 0
        cv2.rectangle(frame, (0, bar_y), (bar_width, bar_y + bar_height), phase_color, -1)
        
        # Draw event markers
        for i, event_frame in enumerate(event_frames):
            if event_frame < total_frames:
                marker_x = int((event_frame / total_frames) * width) if total_frames > 0 else 0
                marker_color = phase_colors[i]
                cv2.line(frame, (marker_x, 0), (marker_x, height), marker_color, 2)
                
                # Label event at marker
                if abs(frame_num - event_frame) < 5:  # Show label near event
                    event_name = EVENT_NAMES[i]
                    conf = event_confidences[i]
                    label = f"{event_name} ({conf:.2f})"
                    cv2.putText(frame, label, (marker_x + 5, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker_color, 2)
        
        # Draw frame number and time
        time_sec = frame_num / fps if fps > 0 else 0
        info_text = f"Frame: {frame_num}/{total_frames} | Time: {time_sec:.2f}s"
        cv2.putText(frame, info_text, (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_num += 1
        
        if frame_num % 50 == 0:
            print(f"  Annotated {frame_num}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"✅ Annotated video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze golf swing video with pose-based cropping and event detection')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, default='models/swingnet_2000.pth.tar', 
                       help='Path to event detector model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: <input>_analyzed.mp4)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run model on')
    parser.add_argument('--buffer-before', type=int, default=75,
                       help='Frames to add before detected swing start')
    parser.add_argument('--buffer-after', type=int, default=75,
                       help='Frames to add after detected swing end')
    parser.add_argument('--skip-crop', action='store_true',
                       help='Skip cropping step (use full video)')
    parser.add_argument('--detect-slowmo', action='store_true',
                       help='Detect and speed up slow-motion regions before event detection')
    parser.add_argument('--slowmo-speed-factor', type=float, default=2.0,
                       help='Speed multiplier for slow-motion regions (default: 2.0 = 2x speed)')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        base_name = osp.splitext(args.video)[0]
        args.output = f"{base_name}_analyzed.mp4"
    
    # Step 1: Crop video using pose-based detection
    cropped_video = None
    if not args.skip_crop:
        print(f"\n{'='*60}")
        print("Step 1: Detecting swing boundaries (pose-based)")
        print(f"{'='*60}")
        
        detector = SwingBoundaryDetector(
            buffer_before=args.buffer_before,
            buffer_after=args.buffer_after,
            subsample_rate=2
        )
        
        try:
            start_frame, end_frame, confidence = detector.detect_swing_boundaries(args.video)
            print(f"✅ Detected swing: frames {start_frame}-{end_frame} (confidence: {confidence:.3f})")
            
            # Crop video
            cropped_video = f"{osp.splitext(args.video)[0]}_cropped_temp.mp4"
            crop_video(args.video, start_frame, end_frame, cropped_video)
            
            # Store swing boundaries for slow-mo detection
            swing_start = start_frame
            swing_end = end_frame
        except Exception as e:
            print(f"⚠️  Pose-based detection failed: {e}")
            print("   Using full video instead")
            cropped_video = args.video
            swing_start = 0
            swing_end = None
    else:
        print("Skipping cropping step (using full video)")
        cropped_video = args.video
        swing_start = 0
        swing_end = None
    
    # Step 1.5: Detect and speed up slow-motion (if requested)
    if args.detect_slowmo:
        try:
            is_slowmo, confidence, speed_factor = detect_slowmo_by_peak_sharpness(
                cropped_video, swing_start_frame=0, swing_end_frame=None
            )
            
            if is_slowmo and confidence > 0.3:
                print(f"\n{'='*60}")
                print(f"⚠️  Slow-motion detected! Speeding up by {speed_factor:.2f}x...")
                print(f"{'='*60}")
                
                # Use user-specified speed factor if provided, otherwise use detected
                actual_speed_factor = speed_factor if args.slowmo_speed_factor == 2.0 else args.slowmo_speed_factor
                
                speeded_video = cropped_video.replace('.mp4', '_speeded.mp4')
                speed_up_slow_motion(cropped_video, speeded_video, speed_factor=actual_speed_factor)
                
                # Use speeded video for event detection
                cropped_video = speeded_video
                print(f"✅ Using speeded-up video for event detection")
            else:
                print(f"✅ No slow-motion detected (confidence: {confidence:.2f})")
        except Exception as e:
            print(f"⚠️  Slow-motion detection failed: {e}")
            print("   Continuing with original video")
    
    # Step 2: Load event detector
    print(f"\n{'='*60}")
    print("Step 2: Loading event detector")
    print(f"{'='*60}")
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = load_event_detector(args.model, device)
    
    # Step 3: Predict events
    print(f"\n{'='*60}")
    print("Step 3: Predicting swing events")
    print(f"{'='*60}")
    
    event_frames, event_confidences, all_probs = predict_events(model, cropped_video, device)
    
    # Create probability graph
    graph_path = osp.splitext(args.output)[0] + '_probabilities.png'
    plot_event_probabilities(all_probs, event_frames, event_confidences, graph_path)
    
    # Step 4: Create annotated video
    print(f"\n{'='*60}")
    print("Step 4: Creating annotated video")
    print(f"{'='*60}")
    
    annotate_video(cropped_video, event_frames, event_confidences, args.output)
    
    # Cleanup temporary files
    temp_cropped = f"{osp.splitext(args.video)[0]}_cropped_temp.mp4"
    temp_speeded = temp_cropped.replace('.mp4', '_speeded.mp4')
    
    if osp.exists(temp_cropped) and temp_cropped != args.video:
        os.remove(temp_cropped)
        print(f"🧹 Cleaned up temporary cropped video")
    
    if osp.exists(temp_speeded):
        os.remove(temp_speeded)
        print(f"🧹 Cleaned up temporary speeded video")
    
    print(f"\n{'='*60}")
    print("✅ Analysis complete!")
    print(f"{'='*60}")
    print(f"Input video: {args.video}")
    print(f"Output video: {args.output}")
    print(f"\nDetected events:")
    for i, (frame, conf) in enumerate(zip(event_frames, event_confidences)):
        print(f"  {EVENT_NAMES[i]:15s}: Frame {frame:4d} (confidence: {conf:.3f})")


if __name__ == '__main__':
    main()
