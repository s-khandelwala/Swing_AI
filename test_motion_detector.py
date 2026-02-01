"""
Standalone test script for motion-based golf swing boundary detection.

This script allows you to test the SwingBoundaryDetector independently
without running the full training pipeline.
"""

import cv2
import numpy as np
import os
import os.path as osp
import sys
from motion_swing_detector import SwingBoundaryDetector

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - motion graphs will be skipped")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - motion graphs will be skipped")


def crop_video(video_path, start_frame, end_frame, output_path):
    """
    Crop video to specified frame range.
    
    Args:
        video_path: Input video path
        start_frame: Start frame (inclusive)
        end_frame: End frame (inclusive)
        output_path: Output video path
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    frames_written = 0
    
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frames_written += 1
        frame_count += 1
    
    cap.release()
    out.release()
    
    return frames_written


def plot_motion_graph(video_path, detector, start_frame, end_frame, fps, total_frames):
    """
    Plot motion graphs showing detected swing boundaries.
    Generates TWO graphs: one with boosted motion (used for detection) and one without.
    
    Args:
        video_path: Path to video file
        detector: SwingBoundaryDetector instance
        start_frame: Detected start frame
        end_frame: Detected end frame
        fps: Video FPS
        total_frames: Total frames in video
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        # Use stored motion data from detector (if available)
        if hasattr(detector, '_last_motion_data') and detector._last_motion_data:
            motion_data = detector._last_motion_data
            frame_indices = motion_data['frame_indices']
            smoothed_boosted = motion_data['smoothed_boosted']
            smoothed_unboosted = motion_data['smoothed_unboosted']
            raw_boosted = motion_data['raw_boosted']
            raw_unboosted = motion_data['raw_unboosted']
            slow_motion_regions = motion_data.get('slow_motion_regions', [])
            stored_fps = motion_data.get('fps', fps)
        else:
            print("   ⚠️  No stored motion data, recomputing...")
            return  # Can't generate graphs without stored data
        
        video_dir = osp.dirname(video_path)
        video_name = osp.splitext(osp.basename(video_path))[0]
        time_axis = np.array(frame_indices) / stored_fps
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Graph 1: BOOSTED motion (used for detection)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis, smoothed_boosted, 'b-', linewidth=2, label='Motion (Boosted)')
        ax.plot(time_axis, raw_boosted, 'c--', alpha=0.5, linewidth=1, label='Raw Motion (Boosted)')
        
        # Mark slow motion regions
        if slow_motion_regions:
            for slow_start, slow_end in slow_motion_regions:
                if slow_start < len(time_axis) and slow_end < len(time_axis):
                    slow_start_time = time_axis[slow_start]
                    slow_end_time = time_axis[min(slow_end, len(time_axis)-1)]
                    ax.axvspan(slow_start_time, slow_end_time, alpha=0.1, color='purple', label='Slow Motion Region' if slow_start == slow_motion_regions[0][0] else '')
        
        # Mark peak
        peak_idx = np.argmax(smoothed_boosted)
        peak_time = time_axis[peak_idx]
        peak_value = smoothed_boosted[peak_idx]
        ax.plot(peak_time, peak_value, 'ro', markersize=10, label=f'Peak Motion ({peak_time:.2f}s)')
        
        # Mark detected boundaries
        ax.axvline(start_time, color='g', linestyle='--', linewidth=2, label=f'Start ({start_time:.2f}s)')
        ax.axvline(end_time, color='r', linestyle='--', linewidth=2, label=f'End ({end_time:.2f}s)')
        ax.axvspan(start_time, end_time, alpha=0.2, color='yellow', label='Detected Swing')
        
        # Add threshold line
        baseline = np.median(smoothed_boosted)
        peak_motion = smoothed_boosted[peak_idx]
        motion_range = peak_motion - baseline
        threshold = baseline + detector.start_threshold_ratio * motion_range
        ax.axhline(threshold, color='orange', linestyle=':', linewidth=1, label=f'Start Threshold ({detector.start_threshold_ratio*100:.0f}%)')
        ax.axhline(baseline, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Baseline')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Motion Magnitude (normalized, BOOSTED)', fontsize=12)
        ax.set_title(f'Motion Analysis (BOOSTED - Used for Detection): {osp.basename(video_path)}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plot_path_boosted = osp.join(video_dir, f"{video_name}_motion_graph_boosted.png")
        plt.tight_layout()
        plt.savefig(plot_path_boosted, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Boosted motion graph saved: {plot_path_boosted}")
        
        # Graph 2: UNBOOSTED motion (original values)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis, smoothed_unboosted, 'b-', linewidth=2, label='Motion (Unboosted)')
        ax.plot(time_axis, raw_unboosted, 'c--', alpha=0.5, linewidth=1, label='Raw Motion (Unboosted)')
        
        # Mark slow motion regions
        if slow_motion_regions:
            for slow_start, slow_end in slow_motion_regions:
                if slow_start < len(time_axis) and slow_end < len(time_axis):
                    slow_start_time = time_axis[slow_start]
                    slow_end_time = time_axis[min(slow_end, len(time_axis)-1)]
                    ax.axvspan(slow_start_time, slow_end_time, alpha=0.1, color='purple', label='Slow Motion Region' if slow_start == slow_motion_regions[0][0] else '')
        
        # Mark peak (using boosted peak for reference)
        peak_idx_boosted = np.argmax(smoothed_boosted)
        peak_time_boosted = time_axis[peak_idx_boosted]
        peak_value_unboosted = smoothed_unboosted[peak_idx_boosted]
        ax.plot(peak_time_boosted, peak_value_unboosted, 'ro', markersize=10, label=f'Peak (from boosted) ({peak_time_boosted:.2f}s)')
        
        # Also mark unboosted peak
        peak_idx_unboosted = np.argmax(smoothed_unboosted)
        peak_time_unboosted = time_axis[peak_idx_unboosted]
        peak_value_unboosted_peak = smoothed_unboosted[peak_idx_unboosted]
        ax.plot(peak_time_unboosted, peak_value_unboosted_peak, 'mo', markersize=8, label=f'Peak (unboosted) ({peak_time_unboosted:.2f}s)')
        
        # Mark detected boundaries
        ax.axvline(start_time, color='g', linestyle='--', linewidth=2, label=f'Start ({start_time:.2f}s)')
        ax.axvline(end_time, color='r', linestyle='--', linewidth=2, label=f'End ({end_time:.2f}s)')
        ax.axvspan(start_time, end_time, alpha=0.2, color='yellow', label='Detected Swing')
        
        # Add threshold line (using boosted threshold for reference)
        baseline_unboosted = np.median(smoothed_unboosted)
        peak_motion_unboosted = smoothed_unboosted[peak_idx_boosted]  # Use boosted peak index
        motion_range_unboosted = peak_motion_unboosted - baseline_unboosted
        threshold_unboosted = baseline_unboosted + detector.start_threshold_ratio * motion_range_unboosted
        ax.axhline(threshold_unboosted, color='orange', linestyle=':', linewidth=1, label=f'Start Threshold ({detector.start_threshold_ratio*100:.0f}%)')
        ax.axhline(baseline_unboosted, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Baseline')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Motion Magnitude (normalized, UNBOOSTED)', fontsize=12)
        ax.set_title(f'Motion Analysis (UNBOOSTED - Original Values): {osp.basename(video_path)}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plot_path_unboosted = osp.join(video_dir, f"{video_name}_motion_graph_unboosted.png")
        plt.tight_layout()
        plt.savefig(plot_path_unboosted, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Unboosted motion graph saved: {plot_path_unboosted}")
        
    except Exception as e:
        print(f"   ⚠️  Error generating motion graphs: {e}")
        import traceback
        traceback.print_exc()


def test_detector_on_video(video_path, detector, output_path=None):
    """
    Test the detector on a single video and crop it.
    
    Args:
        video_path: Path to video file
        detector: SwingBoundaryDetector instance
        output_path: Path to save cropped video (None = auto-generate)
        
    Returns:
        Path to cropped video, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Processing: {osp.basename(video_path)}")
    print(f"{'='*60}")
    
    # Check if video exists
    if not osp.exists(video_path):
        print(f"❌ Error: Video not found: {video_path}")
        return None
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Original video: {total_frames} frames, {total_frames/fps:.2f}s, {width}x{height}")
    
    # Detect boundaries
    print(f"🔍 Detecting swing boundaries...")
    if detector.skip_early_percent > 0:
        print(f"   Skip early: {detector.skip_early_percent}% ({int(total_frames * detector.skip_early_percent / 100)} frames)")
    else:
        print(f"   Skip early: None (detecting from start)")
    print(f"   Start threshold: {detector.start_threshold_ratio*100:.0f}% of way to peak")
    try:
        start_frame, end_frame, confidence = detector.detect_swing_boundaries(video_path)
        
        # Generate motion graph
        if MATPLOTLIB_AVAILABLE:
            try:
                plot_motion_graph(video_path, detector, start_frame, end_frame, fps, total_frames)
            except Exception as e:
                print(f"   ⚠️  Could not generate motion graph: {e}")
        
        cropped_frames = end_frame - start_frame + 1
        print(f"✅ Detected swing: frames {start_frame}-{end_frame} ({cropped_frames} frames, {cropped_frames/fps:.2f}s)")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Start time: {start_frame/fps:.2f}s, End time: {end_frame/fps:.2f}s")
        
        # Check if start is at frame 0 (problem!)
        if start_frame == 0:
            print(f"   ❌ ERROR: Start frame is 0! Detection failed or video has motion from start.")
            print(f"   💡 Try increasing --skip-early-percent (current: {detector.skip_early_percent}%)")
            print(f"   💡 Or try increasing --start-threshold-ratio (current: {detector.start_threshold_ratio})")
        
        # Warn if detected range seems too small or too large
        crop_ratio = cropped_frames / total_frames
        if crop_ratio < 0.1:
            print(f"   ⚠️  Warning: Detected range is very small ({crop_ratio*100:.1f}% of video)")
        elif crop_ratio > 0.9:
            print(f"   ⚠️  Warning: Detected range is very large ({crop_ratio*100:.1f}% of video)")
        
        # Warn if start is very early or end is very late
        if start_frame < total_frames * 0.05:
            print(f"   ⚠️  Warning: Start frame is very early ({start_frame/fps:.2f}s, {start_frame/total_frames*100:.1f}% into video)")
        if end_frame > total_frames * 0.95:
            print(f"   ⚠️  Warning: End frame is very late ({end_frame/fps:.2f}s)")
        
        # Generate output path if not provided
        if output_path is None:
            video_dir = osp.dirname(video_path)
            video_name = osp.splitext(osp.basename(video_path))[0]
            output_path = osp.join(video_dir, f"{video_name}_cropped.mp4")
        
        # Crop video
        print(f"✂️  Cropping video...")
        frames_written = crop_video(video_path, start_frame, end_frame, output_path)
        
        print(f"✅ Cropped video saved: {output_path}")
        print(f"   Frames: {frames_written} ({frames_written/fps:.2f}s)")
        print(f"   Reduction: {100*(1 - frames_written/total_frames):.1f}% removed")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_multiple_videos(video_dir, detector, output_dir=None, num_videos=None):
    """
    Process multiple videos and crop them.
    
    Args:
        video_dir: Directory containing videos
        detector: SwingBoundaryDetector instance
        output_dir: Directory to save cropped videos (None = same as input)
        num_videos: Number of videos to process (None = all)
    """
    print(f"\n{'='*60}")
    print(f"Processing videos from: {video_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Find video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend([f for f in os.listdir(video_dir) if f.endswith(ext)])
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    video_files = sorted(video_files)
    if num_videos:
        video_files = video_files[:num_videos]
    
    print(f"Found {len(video_files)} video(s) to process\n")
    
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing {video_file}...")
        video_path = osp.join(video_dir, video_file)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            video_name = osp.splitext(video_file)[0]
            output_path = osp.join(output_dir, f"{video_name}_cropped.mp4")
        else:
            output_path = None
        
        result = test_detector_on_video(video_path, detector, output_path)
        if result:
            successful += 1
        else:
            failed += 1
        print()
    
    print(f"{'='*60}")
    print(f"Processing complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test motion-based golf swing boundary detector')
    parser.add_argument('--video', type=str, help='Path to single video file to crop')
    parser.add_argument('--video-dir', type=str, help='Directory containing videos to crop')
    parser.add_argument('--output', type=str, default=None, help='Output video path (for single video) or directory (for multiple videos)')
    parser.add_argument('--num-videos', type=int, default=None, help='Number of videos to process (if using --video-dir, None = all)')
    parser.add_argument('--subsample-rate', type=int, default=2, help='Subsampling rate for detection (default: 2)')
    parser.add_argument('--buffer-before', type=int, default=75, help='Frames before detected swing (default: 75)')
    parser.add_argument('--buffer-after', type=int, default=75, help='Frames after detected swing (default: 75)')
    parser.add_argument('--skip-early-percent', type=float, default=0, help='Skip first N%% of video for setup time (default: 0 = no skip, detect from start)')
    parser.add_argument('--start-threshold-ratio', type=float, default=0.30, help='Start threshold as ratio from baseline to peak, 0.0-1.0 (default: 0.30, higher=more conservative, catches later)')
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory to cache detection results')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached results before processing')
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache and args.cache_dir and osp.exists(args.cache_dir):
        import shutil
        print(f"🗑️  Clearing cache directory: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create detector with specified parameters
    detector = SwingBoundaryDetector(
        method='hybrid',
        subsample_rate=args.subsample_rate,
        buffer_before=args.buffer_before,
        buffer_after=args.buffer_after,
        skip_early_percent=args.skip_early_percent,
        start_threshold_ratio=args.start_threshold_ratio,
        cache_dir=args.cache_dir
    )
    
    print(f"🔧 Detector Configuration:")
    print(f"  Method: hybrid")
    print(f"  Subsampling: {args.subsample_rate}x")
    print(f"  Buffer before: {args.buffer_before} frames")
    print(f"  Buffer after: {args.buffer_after} frames")
    if args.skip_early_percent > 0:
        print(f"  Skip early: {args.skip_early_percent}% of video")
    else:
        print(f"  Skip early: None (detecting from start)")
    print(f"  Start threshold: {args.start_threshold_ratio*100:.0f}% of way to peak (higher = catches later)")
    if args.cache_dir:
        print(f"  Cache directory: {args.cache_dir}")
    
    # Process single video or multiple videos
    if args.video:
        output_path = args.output if args.output else None
        test_detector_on_video(args.video, detector, output_path)
    elif args.video_dir:
        output_dir = args.output if args.output else None
        process_multiple_videos(args.video_dir, detector, output_dir, args.num_videos)
    else:
        # Default: process Kaggle dataset if available
        default_video_dir = '/kaggle/input/swingai-model/golfdb-data/data/videos_160'
        if osp.exists(default_video_dir):
            print(f"Using default video directory: {default_video_dir}")
            output_dir = args.output if args.output else None
            process_multiple_videos(default_video_dir, detector, output_dir, args.num_videos)
        else:
            print("❌ Error: Please specify --video or --video-dir")
            print("\nUsage examples:")
            print("  python test_motion_detector.py --video path/to/video.mp4")
            print("  python test_motion_detector.py --video path/to/video.mp4 --output cropped.mp4")
            print("  python test_motion_detector.py --video-dir path/to/videos --output-dir ./cropped")
            print("  python test_motion_detector.py --video-dir path/to/videos --num-videos 10")
            sys.exit(1)
