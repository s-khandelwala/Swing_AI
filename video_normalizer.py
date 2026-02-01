"""
Video Normalization Utility

Normalizes all videos to the same quality and size for consistent training.
This removes frame quality as a variable and ensures all videos are processed equally.

Features:
- Size normalization (resize to target size)
- Quality enhancement (sharpness, contrast, denoising)
- Brightness/contrast normalization
- Frame rate normalization (optional)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd


class VideoNormalizer:
    """
    Normalizes videos to consistent quality and size.
    """
    
    def __init__(
        self,
        target_size=(160, 160),
        enhance_quality=True,
        normalize_brightness=True,
        denoise=True,
        sharpen=True
    ):
        """
        Args:
            target_size: Target (width, height) for all videos
            enhance_quality: Apply quality enhancement (sharpness, contrast)
            normalize_brightness: Normalize brightness/contrast across videos
            denoise: Apply denoising filter
            sharpen: Apply sharpening filter
        """
        self.target_size = target_size
        self.enhance_quality = enhance_quality
        self.normalize_brightness = normalize_brightness
        self.denoise = denoise
        self.sharpen = sharpen
    
    def enhance_frame(self, frame):
        """
        Enhance single frame quality.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Denoise
        if self.denoise:
            l = cv2.fastNlMeansDenoising(l, None, 10, 7, 21)
        
        # Normalize brightness/contrast
        if self.normalize_brightness:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        
        # Sharpen
        if self.sharpen:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            l = cv2.filter2D(l, -1, kernel)
            # Clip to valid range
            l = np.clip(l, 0, 255)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def normalize_frame(self, frame):
        """
        Normalize single frame: resize and enhance.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Normalized frame
        """
        # Resize to target size
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Enhance quality
        if self.enhance_quality:
            frame = self.enhance_frame(frame)
        
        return frame
    
    def normalize_video(self, input_path, output_path, target_fps=None):
        """
        Normalize entire video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to save normalized video
            target_fps: Target FPS (None = keep original)
        
        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if target_fps is None:
            target_fps = fps
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, self.target_size)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Normalize frame
            normalized_frame = self.normalize_frame(frame)
            
            # Write to output
            out.write(normalized_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        return True
    
    def normalize_directory(self, input_dir, output_dir, pattern='*.mp4'):
        """
        Normalize all videos in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            pattern: File pattern to match (default: '*.mp4')
        
        Returns:
            Number of videos processed
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = list(input_path.glob(pattern))
        
        print(f"Found {len(video_files)} videos to normalize")
        print(f"Target size: {self.target_size}")
        print(f"Quality enhancement: {self.enhance_quality}")
        
        processed = 0
        failed = 0
        
        for video_file in tqdm(video_files, desc="Normalizing videos"):
            output_file = output_path / video_file.name
            
            # Skip if already processed
            if output_file.exists():
                continue
            
            try:
                success = self.normalize_video(str(video_file), str(output_file))
                if success:
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\nError processing {video_file.name}: {e}")
                failed += 1
        
        print(f"\nNormalization complete:")
        print(f"  Processed: {processed}")
        print(f"  Failed: {failed}")
        print(f"  Output directory: {output_dir}")
        
        return processed


def normalize_golfdb_videos(
    input_dir='data/videos_160/',
    output_dir='data/videos_normalized/',
    target_size=(160, 160)
):
    """
    Normalize all GolfDB videos to consistent quality and size.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save normalized videos
        target_size: Target (width, height) - should match existing size
    """
    normalizer = VideoNormalizer(
        target_size=target_size,
        enhance_quality=True,
        normalize_brightness=True,
        denoise=True,
        sharpen=True
    )
    
    # Normalize all videos
    processed = normalizer.normalize_directory(input_dir, output_dir)
    
    print(f"\nNormalized {processed} videos")
    print(f"All videos now have:")
    print(f"  - Size: {target_size}")
    print(f"  - Enhanced quality (sharpness, contrast, denoising)")
    print(f"  - Normalized brightness/contrast")
    print(f"\nUpdate your dataloader to use: {output_dir}")


def normalize_single_video(input_path, output_path, target_size=(160, 160)):
    """
    Normalize a single video file.
    
    Args:
        input_path: Path to input video
        output_path: Path to save normalized video
        target_size: Target (width, height)
    """
    normalizer = VideoNormalizer(
        target_size=target_size,
        enhance_quality=True,
        normalize_brightness=True,
        denoise=True,
        sharpen=True
    )
    
    success = normalizer.normalize_video(input_path, output_path)
    
    if success:
        print(f"Normalized video saved to: {output_path}")
    else:
        print(f"Failed to normalize video: {input_path}")
    
    return success


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize golf swing videos')
    parser.add_argument('--input-dir', type=str, default='data/videos_160/',
                       help='Input directory containing videos')
    parser.add_argument('--output-dir', type=str, default='data/videos_normalized/',
                       help='Output directory for normalized videos')
    parser.add_argument('--target-size', type=int, nargs=2, default=[160, 160],
                       help='Target size (width height)')
    parser.add_argument('--single', type=str, default=None,
                       help='Normalize single video file (provide path)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single video')
    
    args = parser.parse_args()
    
    if args.single:
        # Normalize single video
        if not args.output:
            args.output = args.single.replace('.mp4', '_normalized.mp4')
        normalize_single_video(args.single, args.output, tuple(args.target_size))
    else:
        # Normalize directory
        normalize_golfdb_videos(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=tuple(args.target_size)
        )

