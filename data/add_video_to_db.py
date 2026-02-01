import pandas as pd
import numpy as np
import os
import shutil
import sys

# Add current directory to path to import preprocess_videos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_videos import preprocess_videos

def add_video_to_database(video_path, events, bbox=None, output_name=None, 
                          player="Unknown", sex="M", club="driver", 
                          view="dtl", slow=0, split=1, dim=160):
    """
    Add a new video to the golfDB database.
    
    Args:
        video_path: Path to input video file
        events: List of 8 event frame numbers [Address, Toe-up, Mid-backswing, Top, 
                Mid-downswing, Impact, Mid-follow-through, Finish]
        bbox: Bounding box as [x, y, w, h] normalized (0-1). If None, uses full frame [0, 0, 1, 1]
        output_name: Name for the video (default: basename without extension)
        player: Player name
        sex: "M" or "F"
        club: Club type (e.g., "driver", "iron", "putter)
        view: Camera view (e.g., "dtl" for down-the-line, "face" for face-on)
        slow: 0 for normal speed, 1 for slow motion
        split: Cross-validation split (1-4)
        dim: Output video dimension (default 160)
    """
    # Load database
    db_path = 'golfDB.pkl'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    df = pd.read_pickle(db_path)
    
    # Validate events
    if len(events) != 8:
        raise ValueError(f"Expected 8 events, got {len(events)}")
    
    # Generate new ID (max existing ID + 1)
    new_id = int(df['id'].max()) + 1 if len(df) > 0 else 1
    
    # Convert events to numpy array
    events_array = np.array(events)
    
    # Set default bbox if not provided (full frame)
    if bbox is None:
        bbox = np.array([0.0, 0.0, 1.0, 1.0])  # [x, y, w, h] normalized
    else:
        bbox = np.array(bbox)
    
    # Generate youtube_id (use output_name or basename)
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Remove any file extension to avoid double .mp4
    youtube_id = os.path.splitext(output_name)[0]
    
    # Create new row
    new_row = {
        'id': new_id,
        'youtube_id': youtube_id,
        'player': player,
        'sex': sex,
        'club': club,
        'view': view,
        'slow': slow,
        'events': events_array,
        'bbox': bbox,
        'split': split
    }
    
    # Add to dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save database
    df.to_pickle(db_path)
    print(f"\n✓ Added video to database with ID: {new_id}")
    print(f"  Events: {events}")
    print(f"  Bbox: {bbox}")
    
    # Copy video to expected location for preprocessing
    yt_video_dir = '../../database/videos/'
    os.makedirs(yt_video_dir, exist_ok=True)
    dest_path = os.path.join(yt_video_dir, f'{youtube_id}.mp4')
    
    if not os.path.exists(dest_path):
        shutil.copy2(video_path, dest_path)
        print(f"✓ Copied video to: {dest_path}")
    else:
        print(f"⚠ Video already exists at: {dest_path}")
    
    # Preprocess the video using existing function
    print(f"\nPreprocessing video {new_id}...")
    preprocess_videos(new_id, dim=dim)
    
    return new_id

def parse_events(events_str):
    """Parse events string into list of integers."""
    try:
        events = [int(x.strip()) for x in events_str.split()]
        if len(events) != 8:
            raise ValueError(f"Expected 8 events, got {len(events)}")
        return events
    except ValueError as e:
        raise ValueError(f"Invalid events format: {e}")

def parse_bbox(bbox_str):
    """Parse bbox string into list of floats."""
    if not bbox_str.strip():
        return None
    try:
        bbox = [float(x.strip()) for x in bbox_str.split()]
        if len(bbox) != 4:
            raise ValueError(f"Expected 4 values (x y w h), got {len(bbox)}")
        if not all(0 <= x <= 1 for x in bbox):
            print("⚠ Warning: Bbox values should be normalized (0-1)")
        return bbox
    except ValueError as e:
        raise ValueError(f"Invalid bbox format: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("Add Video to GolfDB Database")
    print("=" * 60)
    print()
    
    # Get video path
    video_path = input("Enter path to video file: ").strip().strip('"\'')
    if not os.path.exists(video_path):
        print(f"✗ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"\n✓ Found video: {video_path}")
    print()
    
    # Get events
    print("Enter 8 event frame numbers (space-separated):")
    print("  Format: Address Toe-up Mid-backswing Top Mid-downswing Impact Mid-follow-through Finish")
    print("  Example: 10 25 40 55 70 85 100 120")
    events_str = input("Events: ").strip()
    try:
        events = parse_events(events_str)
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    print(f"✓ Parsed events: {events}")
    print()
    
    # Get bbox (optional)
    print("Enter bounding box (normalized 0-1) as [x y w h] or press Enter for full frame:")
    print("  Example: 0.2 0.1 0.6 0.8")
    bbox_str = input("Bbox (optional): ").strip()
    try:
        bbox = parse_bbox(bbox_str)
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    if bbox is None:
        print("✓ Using full frame [0, 0, 1, 1]")
    else:
        print(f"✓ Using bbox: {bbox}")
    print()
    
    # Get output name (optional)
    output_name = input("Enter output name (or press Enter to use video filename): ").strip()
    if not output_name:
        output_name = None
        print("✓ Using video filename")
    else:
        print(f"✓ Output name: {output_name}")
    print()
    
    # Get player info
    player = input("Enter player name (default: Unknown): ").strip()
    if not player:
        player = "Unknown"
    print(f"✓ Player: {player}")
    print()
    
    # Get sex
    sex = input("Enter sex (M/F, default: M): ").strip().upper()
    if sex not in ['M', 'F']:
        sex = 'M'
    print(f"✓ Sex: {sex}")
    print()
    
    # Get club
    club = input("Enter club type (default: driver): ").strip()
    if not club:
        club = "driver"
    print(f"✓ Club: {club}")
    print()
    
    # Get view
    view = input("Enter camera view (default: dtl): ").strip()
    if not view:
        view = "dtl"
    print(f"✓ View: {view}")
    print()
    
    # Get slow motion flag
    slow_str = input("Is this slow motion? (y/n, default: n): ").strip().lower()
    slow = 1 if slow_str in ['y', 'yes'] else 0
    print(f"✓ Slow motion: {slow}")
    print()
    
    # Get split
    split_str = input("Enter cross-validation split (1-4, default: 1): ").strip()
    try:
        split = int(split_str) if split_str else 1
        if split not in [1, 2, 3, 4]:
            raise ValueError()
    except ValueError:
        split = 1
        print("⚠ Invalid split, using default: 1")
    print(f"✓ Split: {split}")
    print()
    
    # Get dimension
    dim_str = input("Enter output video dimension (default: 160): ").strip()
    try:
        dim = int(dim_str) if dim_str else 160
    except ValueError:
        dim = 160
        print("⚠ Invalid dimension, using default: 160")
    print(f"✓ Dimension: {dim}")
    print()
    
    # Confirm
    print("=" * 60)
    print("Summary:")
    print(f"  Video: {video_path}")
    print(f"  Events: {events}")
    print(f"  Bbox: {bbox if bbox is not None else [0, 0, 1, 1]}")
    print(f"  Output name: {output_name or os.path.splitext(os.path.basename(video_path))[0]}")
    print(f"  Player: {player}")
    print(f"  Sex: {sex}")
    print(f"  Club: {club}")
    print(f"  View: {view}")
    print(f"  Slow motion: {slow}")
    print(f"  Split: {split}")
    print(f"  Dimension: {dim}")
    print("=" * 60)
    print()
    
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)
    
    print()
    
    # Add to database
    try:
        video_id = add_video_to_database(
            video_path,
            events=events,
            bbox=bbox,
            output_name=output_name,
            player=player,
            sex=sex,
            club=club,
            view=view,
            slow=slow,
            split=split,
            dim=dim
        )
        print(f"\n{'=' * 60}")
        print(f"✓ Successfully added and preprocessed video with ID: {video_id}")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)