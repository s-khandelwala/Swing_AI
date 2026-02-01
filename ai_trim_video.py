# ai_trim_video.py
import argparse
import cv2
import numpy as np
import os
import json
import re
from PIL import Image

def extract_key_frames(video_path, num_frames=10):
    """
    Extract evenly spaced frames from video for AI analysis.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    
    frames = []
    frame_numbers = []
    
    # Extract evenly spaced frames
    step = max(1, total_frames // num_frames)
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_numbers.append(i)
            if len(frames) >= num_frames:
                break
    
    cap.release()
    return frames, frame_numbers, fps, duration, total_frames

def analyze_with_gemini(frames, frame_numbers, fps, total_frames):
    """
    Use Google Gemini to analyze frames and find swing start/end.
    """
    try:
        import google.generativeai as genai
        
        # Prepare images - convert numpy arrays to PIL Images
        images = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            images.append(pil_image)
        
        # Create prompt
        prompt = f"""Analyze these frames from a golf swing video. The frames are evenly spaced throughout the video.

The video has {total_frames} total frames. The frame numbers shown are: {frame_numbers}

Identify the ACTUAL swing frames:
1. swing_start_frame: The frame number where the golfer first takes their address position and begins the actual swing
2. swing_end_frame: The frame number where the golfer completes the follow-through (swing is finished)

These should be the actual swing start and end, NOT the trim points.
Make sure that start should be a maximum of 100 and a minimum of 50 frames before the actual swing and end should be a maximum of 100 and a minimum of 50 frames after the golf swing, unles the video starts right before the swing or ends right after the swing. In that case, respond with the frame number of the start or end of the swing.

Respond in JSON format:
{{"swing_start_frame": <frame_number>, "swing_end_frame": <frame_number>, "confidence": <0.0-1.0>}}

If you cannot clearly identify these, respond with null values.

If the video starts right before the swing or ends right after the swing, respond with the frame number of the start or end of the swing."""
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt] + images)
        
        result_text = response.text
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
            return result.get('swing_start_frame'), result.get('swing_end_frame'), result.get('confidence', 0.5)
    except Exception as e:
        print(f"Gemini API error: {e}")
    
    return None, None, None

def calculate_trim_points(swing_start, swing_end, total_frames, max_before=100, max_after=100, min_before=50, min_after=50):
    """
    Calculate trim points based on swing start/end with padding rules.
    
    Rules:
    - Start: swing_start - min(200, available) but at least 100 before
    - End: swing_end + min(200, available) but at least 100 after
    """
    # Calculate available frames before swing start
    available_before = swing_start
    padding_before = min(max_before, available_before)
    padding_before = max(min_before, padding_before)  # At least min_before
    
    # Calculate available frames after swing end
    available_after = total_frames - swing_end - 1
    padding_after = min(max_after, available_after)
    padding_after = max(min_after, padding_after)  # At least min_after
    
    trim_start = max(0, swing_start - padding_before)
    trim_end = min(total_frames, swing_end + padding_after)
    
    return trim_start, trim_end, padding_before, padding_after

def trim_video_from_swing_frames(video_path, swing_start, swing_end, output_path, total_frames):
    """
    Trim video using detected swing start/end with smart padding.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate trim points with padding rules
    trim_start, trim_end, padding_before, padding_after = calculate_trim_points(
        swing_start, swing_end, total_frames
    )
    
    print(f"\nSwing detection:")
    print(f"  Swing starts at frame: {swing_start} ({swing_start/fps:.2f}s)")
    print(f"  Swing ends at frame: {swing_end} ({swing_end/fps:.2f}s)")
    print(f"\nTrimming with padding:")
    print(f"  Padding before: {padding_before} frames (max 100, min 50)")
    print(f"  Padding after: {padding_after} frames (max 100, min 50)")
    print(f"  Trim start: {trim_start} ({trim_start/fps:.2f}s)")
    print(f"  Trim end: {trim_end} ({trim_end/fps:.2f}s)")
    print(f"  Trimmed length: {trim_end - trim_start} frames ({(trim_end - trim_start)/fps:.2f}s)")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, trim_start)
    frame_num = trim_start
    
    print("\nWriting trimmed video...")
    while frame_num < trim_end:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_num += 1
        
        if frame_num % 30 == 0:
            progress = ((frame_num - trim_start) / (trim_end - trim_start)) * 100
            print(f"  Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"Trimmed video saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Trim golf videos using Gemini AI to detect swing start/end',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--video', required=True, help='Path to input video')
    parser.add_argument('-o', '--output', default=None, help='Output video path')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--num-frames', type=int, default=25,
                       help='Number of frames to extract for analysis (default: 10)')
    parser.add_argument('--max-before', type=int, default=100,
                       help='Maximum frames before swing start (default: 100)')
    parser.add_argument('--max-after', type=int, default=100,
                       help='Maximum frames after swing end (default: 100)')
    parser.add_argument('--min-before', type=int, default=50,
                       help='Minimum frames before swing start (default: 50)')
    parser.add_argument('--min-after', type=int, default=50,
                       help='Minimum frames after swing end (default: 50)')
    
    args = parser.parse_args()
    
    # Setup API key
    if args.api_key:
        os.environ['GEMINI_API_KEY'] = args.api_key
    elif 'GEMINI_API_KEY' not in os.environ:
        print("Error: Gemini API key required. Set GEMINI_API_KEY env var or use --api-key")
        return
    
    # Setup output path
    if args.output is None:
        base_name = os.path.splitext(args.video)[0]
        ext = os.path.splitext(args.video)[1]
        args.output = f"{base_name}_trimmed{ext}"
    
    print(f"\n{'='*60}")
    print("AI-POWERED VIDEO TRIMMING (Gemini)")
    print(f"{'='*60}")
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    
    # Extract key frames
    print(f"\nExtracting {args.num_frames} key frames...")
    frames, frame_numbers, fps, duration, total_frames = extract_key_frames(args.video, args.num_frames)
    print(f"Video: {total_frames} frames, {duration:.2f}s ({fps} fps)")
    print(f"Extracted {len(frames)} frames for analysis")
    
    # Analyze with Gemini
    print(f"\nAnalyzing frames with Gemini to detect swing start/end...")
    swing_start, swing_end, confidence = analyze_with_gemini(frames, frame_numbers, fps, total_frames)
    
    if swing_start is None or swing_end is None:
        print("Error: Could not detect swing start/end. Try increasing --num-frames or check API key.")
        return
    
    print(f"\nAI Detection Results:")
    print(f"  Swing start frame: {swing_start} ({swing_start/fps:.2f}s)")
    print(f"  Swing end frame: {swing_end} ({swing_end/fps:.2f}s)")
    print(f"  Confidence: {confidence:.2f}")
    
    # Trim video with smart padding
    trim_video_from_swing_frames(
        args.video, swing_start, swing_end, args.output, total_frames
    )
    
    print(f"\n{'='*60}")
    print("TRIMMING COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {args.output}\n")

if __name__ == '__main__':
    main()