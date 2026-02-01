# trim_video.py
import argparse
import cv2
import numpy as np
import subprocess
import sys
import os
from test_video import event_names

def detect_first_event_with_model(video_path):
    """
    Use the model to detect the first event (Address) frame.
    Returns the frame number of the first event.
    """
    print("Running model to detect first event...")
    result = subprocess.run(
        [sys.executable, 'test_video.py', '-p', video_path],
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    import re
    events_match = re.search(r'Predicted event frames:\s*\[(.*?)\]', output)
    
    if events_match:
        events_str = events_match.group(1)
        events = []
        for part in events_str.split(','):
            for num_str in part.split():
                if num_str.strip():
                    try:
                        events.append(int(num_str.strip()))
                    except ValueError:
                        pass
        
        if len(events) > 0:
            first_event = events[0]  # Address event
            print(f"Detected first event (Address) at frame: {first_event}")
            return first_event
    
    return None

def detect_motion_start(video_path, threshold=5000, lookback_frames=30):
    """
    Detect when significant motion starts in the video.
    Returns the frame number where motion begins.
    
    Args:
        video_path: Path to video
        threshold: Motion threshold (sum of frame differences)
        lookback_frames: How many frames to look back from detected motion
    """
    print("Detecting motion start...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, prev_frame = cap.read()
    if not ret:
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_num = 1
    motion_start = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, gray)
        motion = np.sum(frame_diff)
        
        if motion > threshold and motion_start is None:
            motion_start = max(0, frame_num - lookback_frames)
            print(f"Motion detected starting around frame: {motion_start}")
            break
        
        prev_gray = gray
        frame_num += 1
        
        # Progress update
        if frame_num % 100 == 0:
            print(f"  Checking frame {frame_num}...")
    
    cap.release()
    return motion_start

def trim_video(video_path, start_frame, end_frame=None, output_path=None, padding_frames=30):
    """
    Trim video from start_frame to end_frame (or end of video).
    
    Args:
        video_path: Input video path
        start_frame: Frame to start from (will add padding_frames before)
        end_frame: Frame to end at (None = end of video)
        output_path: Output video path
        padding_frames: Frames to include before start_frame
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate actual start (with padding)
    actual_start = max(0, start_frame - padding_frames)
    actual_end = end_frame if end_frame is not None else total_frames
    
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        ext = os.path.splitext(video_path)[1]
        output_path = f"{base_name}_trimmed{ext}"
    
    print(f"\nTrimming video:")
    print(f"  Original: {total_frames} frames ({total_frames/fps:.2f}s)")
    print(f"  Start frame: {actual_start} (with {padding_frames} frame padding)")
    print(f"  End frame: {actual_end}")
    print(f"  Trimmed: {actual_end - actual_start} frames ({(actual_end - actual_start)/fps:.2f}s)")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
    
    frame_num = actual_start
    frames_written = 0
    
    print("\nWriting trimmed video...")
    while frame_num < actual_end:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frames_written += 1
        frame_num += 1
        
        if frames_written % 30 == 0:
            progress = ((frame_num - actual_start) / (actual_end - actual_start)) * 100
            print(f"  Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"\nDone! Trimmed video saved to: {output_path}")
    print(f"Frames written: {frames_written}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Trim golf swing videos to remove excess time before swing starts')
    parser.add_argument('-v', '--video', required=True, help='Path to input video')
    parser.add_argument('-o', '--output', help='Output video path (default: adds _trimmed to filename)')
    parser.add_argument('--method', choices=['model', 'motion', 'manual'], default='model',
                       help='Method to detect start: model (uses golf model), motion (motion detection), manual (specify frames)')
    parser.add_argument('--start-frame', type=int, help='Manual start frame (for manual method)')
    parser.add_argument('--end-frame', type=int, help='Manual end frame (optional, for manual method)')
    parser.add_argument('--padding', type=int, default=30,
                       help='Frames to include before detected start (default: 30)')
    parser.add_argument('--motion-threshold', type=int, default=5000,
                       help='Motion detection threshold (default: 5000, for motion method)')
    
    args = parser.parse_args()
    
    start_frame = None
    
    if args.method == 'model':
        start_frame = detect_first_event_with_model(args.video)
        if start_frame is None:
            print("Error: Could not detect first event. Try using --method motion or --method manual")
            return
    elif args.method == 'motion':
        start_frame = detect_motion_start(args.video, args.motion_threshold)
        if start_frame is None:
            print("Error: Could not detect motion start. Try using --method model or --method manual")
            return
    elif args.method == 'manual':
        if args.start_frame is None:
            print("Error: --start-frame required when using --method manual")
            return
        start_frame = args.start_frame
    
    if start_frame is None:
        print("Error: Could not determine start frame")
        return
    
    # Trim the video
    trim_video(args.video, start_frame, args.end_frame, args.output, args.padding)

if __name__ == '__main__':
    main()