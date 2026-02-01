# stitch_phases.py
import argparse
import cv2
import numpy as np
from test_video import event_names
import subprocess
import sys
import re
import os

def get_current_phase(frame_num, events):
    """
    Determine which phase a frame belongs to based on event boundaries.
    Each event frame marks the END of that phase.
    Everything before the first event is considered "Address".
    
    Args:
        frame_num: Current frame number
        events: List of event frame numbers (8 events) - these mark where each phase ENDS
    
    Returns:
        phase_index: Index of current phase (0-7)
    """
    # If before first event, treat as Address (phase 0)
    if frame_num < events[0]:
        return 0  # Address phase (ends at events[0])
    
    # Check each phase boundary - each event marks the end of that phase
    for i in range(len(events)):
        if frame_num < events[i]:
            return i  # We're in phase i, which ends at events[i]
    
    # If we've passed all events, we're in the Finish phase
    return 7  # Finish phase

def annotate_video_with_phases(video_path, events, confidences, output_video='annotated_phases.mp4', slow_down_factor=0.5):
    """
    Process the entire video and overlay phase labels in real-time.
    Creates two videos: one normal speed and one slowed down.
    
    Args:
        video_path: Path to input video
        events: List of event frame numbers
        confidences: List of confidence scores
        output_video: Output video filename (base name, will create two versions)
        slow_down_factor: Speed factor for slowed video (0.5 = half speed, 0.25 = quarter speed)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Events: {events}")
    print(f"Confidences: {confidences}")
    
    # Create output filenames
    base_name = os.path.splitext(output_video)[0]
    ext = os.path.splitext(output_video)[1]
    normal_output = f"{base_name}_normal{ext}"
    slow_output = f"{base_name}_slow{ext}"
    
    # Setup video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_normal = cv2.VideoWriter(normal_output, fourcc, fps, (width, height))
    out_slow = cv2.VideoWriter(slow_output, fourcc, int(fps * slow_down_factor), (width, height))
    
    frame_num = 0
    current_phase_idx = -1
    
    print("\nProcessing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Determine current phase
        phase_idx = get_current_phase(frame_num, events)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw phase information
        if phase_idx >= 0 and phase_idx < len(event_names):
            phase_name = event_names[phase_idx]
            confidence = confidences[phase_idx] if phase_idx < len(confidences) else 0.0
            
            # Background rectangle for text (semi-transparent)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # Phase name (large, top left)
            cv2.putText(annotated_frame, phase_name, (30, 50), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
            
            # Confidence score
            conf_text = f'Confidence: {confidence:.3f}'
            cv2.putText(annotated_frame, conf_text, (30, 90), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            
            # Frame number and phase indicator
            frame_text = f'Frame: {frame_num} / {total_frames}'
            cv2.putText(annotated_frame, frame_text, (width - 250, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            
            
            # Progress bar showing phase progression
                        # Progress bar showing phase progression
            if phase_idx < len(events):
                # Calculate progress within current phase
                # Phase starts at the previous event (or 0 for first phase)
                phase_start = events[phase_idx - 1] if phase_idx > 0 else 0
                phase_end = events[phase_idx]
                if phase_end > phase_start:
                    progress = (frame_num - phase_start) / (phase_end - phase_start)
                    progress = max(0, min(1, progress))  # Clamp between 0 and 1
                    bar_width = int((width - 40) * progress)
                    cv2.rectangle(annotated_frame, (20, height - 40), (20 + bar_width, height - 20), 
                                 (0, 255, 0), -1)
                    cv2.rectangle(annotated_frame, (20, height - 40), (width - 20, height - 20), 
                                 (255, 255, 255), 2)
        
        # Mark event frames with a special indicator
        if frame_num in events:
            event_idx = events.index(frame_num)
            cv2.circle(annotated_frame, (width - 50, 50), 20, (0, 0, 255), -1)
            event_name = event_names[event_idx] if event_idx < len(event_names) else 'EVENT'
            cv2.putText(annotated_frame, event_name, (width - 200, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Write to both videos
        out_normal.write(annotated_frame)
        out_slow.write(annotated_frame)
        current_phase_idx = phase_idx
        
        # Progress update
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)")
        
        frame_num += 1
    
    cap.release()
    out_normal.release()
    out_slow.release()
    
    print(f"\nDone! Output saved to:")
    print(f"  Normal speed: {normal_output}")
    print(f"  Slow speed ({slow_down_factor}x): {slow_output}")
    print(f"Total frames processed: {frame_num}")
    
    # Print phase summary
    # Print phase summary
    print("\nPhase Summary:")
    for i, (event_frame, confidence) in enumerate(zip(events, confidences)):
        phase_name = event_names[i]
        # Phase starts at previous event (or 0 for first phase) and ends at current event
        phase_start = events[i - 1] if i > 0 else 0
        phase_end = event_frame
        duration = (phase_end - phase_start) / fps
        print(f"  {phase_name}: Frames {phase_start}-{phase_end}, Duration: {duration:.2f}s, Confidence: {confidence:.3f}")


def detect_events(video_path):
    """
    Run test_video.py to detect events and return events/confidences.
    """
    print("Running detection...")
    result = subprocess.run(
        [sys.executable, 'test_video.py', '-p', video_path],
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    print("Detection output:")
    print(output)
    
    # Parse output to get events and confidences
    events_match = re.search(r'Predicted event frames:\s*\[(.*?)\]', output)
    conf_match = re.search(r'Condifence:\s*\[(.*?)\]', output)
    
    if events_match and conf_match:
        events_str = events_match.group(1)
        conf_str = conf_match.group(1)
        
        # Handle space-separated or comma-separated numbers for events
        events = []
        for part in events_str.split(','):
            for num_str in part.split():
                if num_str.strip():
                    try:
                        events.append(int(num_str.strip()))
                    except ValueError:
                        pass
        
        # Handle np.float32(...) format for confidences
        confidences = []
        # Extract all float values from np.float32(...) format
        np_float_pattern = r'np\.float32\(([\d.]+)\)'
        np_matches = re.findall(np_float_pattern, conf_str)
        for match in np_matches:
            try:
                confidences.append(float(match))
            except ValueError:
                pass
        
        # If that didn't work, try simple float pattern
        if len(confidences) == 0:
            float_pattern = r'(\d+\.\d+)'
            float_matches = re.findall(float_pattern, conf_str)
            for match in float_matches:
                try:
                    confidences.append(float(match))
                except ValueError:
                    pass
        
        print(f"\nDetected events: {events}")
        print(f"Confidences: {confidences}")
        
        if len(events) == 8 and len(confidences) == 8:
            return events, confidences
        else:
            print(f"Warning: Expected 8 events and 8 confidences, got {len(events)} and {len(confidences)}")
            if len(events) == 8 and len(confidences) > 0:
                # If we got some confidences but not 8, try to use what we have
                if len(confidences) < 8:
                    # Pad with last value or 1.0
                    while len(confidences) < 8:
                        confidences.append(confidences[-1] if confidences else 1.0)
                return events, confidences[:8]
            return events, confidences
    else:
        print("Error: Could not parse detection results.")
        print("Looking for patterns:")
        print(f"  Events pattern found: {events_match is not None}")
        print(f"  Confidences pattern found: {conf_match is not None}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Annotate golf swing video with phase labels in real-time')
    parser.add_argument('-v', '--video', required=True, help='Path to input video')
    parser.add_argument('-e', '--events', nargs='+', type=int, 
                       help='Event frame numbers (space-separated). If not provided, will run detection.')
    parser.add_argument('-c', '--confidences', nargs='+', type=float,
                       help='Confidence scores (space-separated). Must match events.')
    parser.add_argument('-o', '--output', default='annotated_phases.mp4', 
                       help='Output video filename (base name, will create _normal and _slow versions)')
    parser.add_argument('--slow-factor', type=float, default=0.5,
                       help='Slow down factor for slowed video (default: 0.5 = half speed)')
    
    args = parser.parse_args()
    
    # Get events and confidences
    if args.events is None:
        events, confidences = detect_events(args.video)
        if events is None or confidences is None:
            print("\nError: Could not detect events. Please provide them manually:")
            print("  python stitch_phases.py -v <video> -e <frame1> <frame2> ... -c <conf1> <conf2> ...")
            return
    else:
        events = args.events
        if args.confidences:
            confidences = args.confidences
        else:
            print("Warning: No confidences provided. Using default value of 1.0")
            confidences = [1.0] * len(events)
    
    if len(events) != len(confidences):
        print(f"Error: Number of events ({len(events)}) doesn't match confidences ({len(confidences)})")
        return
    
    if len(events) != 8:
        print(f"Warning: Expected 8 events, got {len(events)}")
    
    # Annotate video (creates both normal and slow versions)
    annotate_video_with_phases(args.video, events, confidences, args.output, args.slow_factor)


if __name__ == '__main__':
    main()