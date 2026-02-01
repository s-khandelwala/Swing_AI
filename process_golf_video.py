# process_golf_video.py
import argparse
import cv2
import numpy as np
import subprocess
import sys
import os
import re
from test_video import event_names

def detect_events_from_video(video_path, model_path='models/swingnet_2000.pth.tar'):
    """
    Run test_video.py to detect all events and return events/confidences.
    """
    print(f"\n{'='*60}")
    print("Detecting golf swing events...")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, 'test_video.py', '-p', video_path, '-m', model_path],
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    print(output)
    
    # Parse output
    events_match = re.search(r'Predicted event frames:\s*\[(.*?)\]', output)
    conf_match = re.search(r'Confidence:\s*\[(.*?)\]', output)  # Fixed typo: Condifence -> Confidence
    
    if events_match and conf_match:
        events_str = events_match.group(1)
        conf_str = conf_match.group(1)
        
        # Parse events
        events = []
        for part in events_str.split(','):
            for num_str in part.split():
                if num_str.strip():
                    try:
                        events.append(int(num_str.strip()))
                    except ValueError:
                        pass
        
        # Parse confidences
        confidences = []
        np_float_pattern = r'np\.float32\(([\d.]+)\)'
        np_matches = re.findall(np_float_pattern, conf_str)
        for match in np_matches:
            try:
                confidences.append(float(match))
            except ValueError:
                pass
        
        if len(confidences) == 0:
            float_pattern = r'(\d+\.\d+)'
            float_matches = re.findall(float_pattern, conf_str)
            for match in float_matches:
                try:
                    confidences.append(float(match))
                except ValueError:
                    pass
        
        if len(events) == 8 and len(confidences) == 8:
            return events, confidences
    
    return None, None

def trim_video_smart(video_path, events, output_path, padding_before=60, padding_after=30):
    """
    Trim video to keep padding before Address event and after Finish event.
    
    Args:
        video_path: Input video
        events: List of all 8 event frame numbers (from original video)
        output_path: Output video path
        padding_before: Frames to keep before Address (for testing Address detection)
        padding_after: Frames to keep after Finish event
    """
    print(f"\n{'='*60}")
    print("Trimming video...")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate trim points using all events
    address_frame = events[0]  # First event
    finish_frame = events[7]   # Last event (Finish)
    
    start_frame = max(0, address_frame - padding_before)
    end_frame = min(total_frames, finish_frame + padding_after)
    
    print(f"Original video: {total_frames} frames ({total_frames/fps:.2f}s)")
    print(f"Address event at frame: {address_frame}")
    print(f"Finish event at frame: {finish_frame}")
    print(f"Trimming from frame {start_frame} to {end_frame}")
    print(f"  - Keeping {padding_before} frames before Address")
    print(f"  - Keeping {padding_after} frames after Finish")
    print(f"Trimmed length: {end_frame - start_frame} frames ({(end_frame - start_frame)/fps:.2f}s)")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    frames_written = 0
    
    print("\nWriting trimmed video...")
    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frames_written += 1
        frame_num += 1
        
        if frames_written % 30 == 0:
            progress = ((frame_num - start_frame) / (end_frame - start_frame)) * 100
            print(f"  Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"Trimmed video saved to: {output_path}")
    print(f"Frames written: {frames_written} ({(frames_written)/fps:.2f}s)")
    
    return output_path, start_frame

def trim_with_ai(video_path, output_path, api_key=None):
    """
    Use AI (Gemini) to detect swing timeframes and trim video.
    """
    print(f"\n{'='*60}")
    print("Step 0: Using Gemini AI to detect swing timeframes...")
    print(f"{'='*60}")
    
    # Run AI trim script
    cmd = [
        sys.executable, 'ai_trim_video.py',
        '-v', video_path,
        '-o', output_path
    ]
    
    if api_key:
        cmd.extend(['--api-key', api_key])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def create_annotated_videos(video_path, events, confidences, output_base, slow_factor=0.5):
    """
    Create annotated videos using stitch_phases.py logic.
    """
    print(f"\n{'='*60}")
    print("Creating annotated videos...")
    print(f"{'='*60}")
    
    # Import the annotation function from stitch_phases
    try:
        from stitch_phases import annotate_video_with_phases
        annotate_video_with_phases(video_path, events, confidences, output_base, slow_factor)
    except ImportError:
        # If import fails, run stitch_phases.py as subprocess
        print("Running stitch_phases.py...")
        base_name = os.path.splitext(output_base)[0]
        ext = os.path.splitext(output_base)[1]
        normal_output = f"{base_name}_normal{ext}"
        
        # Build command
        events_str = ' '.join(map(str, events))
        confidences_str = ' '.join(map(str, confidences))
        cmd = [
            sys.executable, 'stitch_phases.py',
            '-v', video_path,
            '-e'] + events_str.split() + ['-c'] + confidences_str.split() + [
            '-o', normal_output,
            '--slow-factor', str(slow_factor)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

def main():
    parser = argparse.ArgumentParser(
        description='Complete golf video processing: trim, detect, and annotate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with AI trimming (recommended - faster)
  python process_golf_video.py -v input.mp4 --use-ai
  
  # Process video with model-based trimming
  python process_golf_video.py -v input.mp4
  
  # Custom slow speed and model path
  python process_golf_video.py -v input.mp4 --use-ai --slow-factor 0.25 -m models/swingnet_2000.pth.tar
        """
    )
    parser.add_argument('-v', '--video', required=True, help='Path to input video')
    parser.add_argument('-o', '--output-base', default=None,
                       help='Base name for output files (default: input name)')
    parser.add_argument('-m', '--model', default='models/swingnet_2000.pth.tar',
                       help='Path to model checkpoint (default: models/swingnet_2000.pth.tar)')
    parser.add_argument('--slow-factor', type=float, default=0.5,
                       help='Slow down factor for slow video (default: 0.5 = half speed)')
    parser.add_argument('--skip-trim', action='store_true',
                       help='Skip trimming step (use if video is already trimmed)')
    parser.add_argument('--use-ai', action='store_true',
                       help='Use Gemini AI for trimming (faster and more accurate)')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Setup output names
    if args.output_base is None:
        base_name = os.path.splitext(args.video)[0]
    else:
        base_name = args.output_base
    
    trimmed_video = f"{base_name}_trimmed.mp4"
    output_base = f"{base_name}_annotated.mp4"
    
    print(f"\n{'='*60}")
    print("GOLF VIDEO PROCESSING PIPELINE")
    print(f"{'='*60}")
    print(f"Input video: {args.video}")
    print(f"Output base: {base_name}")
    print(f"Model: {args.model}")
    print(f"Slow factor: {args.slow_factor}x")
    print(f"Trimming method: {'AI (Gemini)' if args.use_ai else 'Model-based'}")
    
    # Step 0: Trim video
    if not args.skip_trim:
        if args.use_ai:
            # Use AI trimming
            success = trim_with_ai(args.video, trimmed_video, args.api_key)
            if not success:
                print("AI trimming failed. Exiting.")
                return
        else:
            # Use model-based trimming
            print(f"\n{'='*60}")
            print("Step 0: Detecting all events in original video...")
            print(f"{'='*60}")
            
            events, confidences = detect_events_from_video(args.video, args.model)
            if events is None:
                print("Error: Could not detect events")
                return
            
            print(f"\nDetected events: {events}")
            for i, event_frame in enumerate(events):
                print(f"  {event_names[i]}: Frame {event_frame}")
            
            # Trim using model detection
            trimmed_path, start_frame = trim_video_smart(
                args.video, events, trimmed_video, 60, 30
            )
            
            # Adjust event frames relative to trimmed video start
            events_trimmed = [e - start_frame for e in events]
            print(f"\nAdjusted events for trimmed video: {events_trimmed}")
    else:
        print("\nSkipping trim step (using original video)")
        trimmed_video = args.video
        events_trimmed = None
    
    # Step 1: Detect events (always use original video for model, not trimmed)
    print(f"\n{'='*60}")
    print("Step 1: Detecting events in original video (full video, no trimming)...")
    print(f"{'='*60}")
    
    # Always detect events on the original video (not trimmed) to ensure model sees full video
    # This ensures the model processes the complete video without any trimming
    events_final, confidences = detect_events_from_video(args.video, args.model)
    
    if events_final is None or confidences is None:
        print("Error: Could not detect events in video.")
        if not args.skip_trim and not args.use_ai and events_trimmed:
            # Fallback to adjusted events from original detection
            print("Using events from original video detection.")
            events_final = events_trimmed
            confidences = [1.0] * 8
        else:
            print("Exiting.")
            return
    
    # If video was trimmed, adjust event frames to match trimmed video for annotation
    # (Events are detected on original video, but annotation happens on trimmed video)
    if not args.skip_trim and trimmed_video != args.video:
        if not args.use_ai and events_trimmed is not None:
            # For model-based trimming, use the pre-calculated adjusted events
            print(f"\nUsing adjusted event frames for trimmed video annotation...")
            events_final = events_trimmed
        elif args.use_ai:
            # For AI trimming, we need to adjust events based on trim start
            # Since we don't have trim_start easily, detect events on trimmed video for annotation
            # But note: model already processed full video above
            print(f"\nNote: Events were detected on full video. For annotation on trimmed video,")
            print(f"      detecting events again on trimmed video for proper frame alignment...")
            events_trimmed_ai, conf_trimmed = detect_events_from_video(trimmed_video, args.model)
            if events_trimmed_ai is not None:
                events_final = events_trimmed_ai
                confidences = conf_trimmed
    
    print(f"\nEvents detected in trimmed video:")
    for i, (event, conf) in enumerate(zip(events_final, confidences)):
        print(f"  {event_names[i]}: Frame {event}, Confidence: {conf:.3f}")
    
    # Step 2: Create annotated videos
    create_annotated_videos(trimmed_video, events_final, confidences, output_base, args.slow_factor)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Trimmed video: {trimmed_video}")
    base_name_out = os.path.splitext(output_base)[0]
    ext = os.path.splitext(output_base)[1]
    print(f"Annotated videos:")
    print(f"  Normal: {base_name_out}_normal{ext}")
    print(f"  Slow: {base_name_out}_slow{ext}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()