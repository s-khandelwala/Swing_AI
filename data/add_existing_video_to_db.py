"""
Add an existing preprocessed video (already in videos_160/) to the database.
This is useful when a video file exists but the database entry is missing.
"""

import pandas as pd
import numpy as np
import os
import sys

def add_existing_video_to_db(video_id, events, bbox=None, 
                             player="Unknown", sex="M", club="driver", 
                             view="dtl", slow=0, split=1, youtube_id=None):
    """
    Add an existing preprocessed video to the database.
    
    Args:
        video_id: The ID to use for this video (must not already exist)
        events: List of 8 event frame numbers [Address, Toe-up, Mid-backswing, Top, 
                Mid-downswing, Impact, Mid-follow-through, Finish]
        bbox: Bounding box as [x, y, w, h] normalized (0-1). If None, uses full frame [0, 0, 1, 1]
        player: Player name
        sex: "M" or "F"
        club: Club type (e.g., "driver", "iron", "putter")
        view: Camera view (e.g., "dtl" for down-the-line, "face" for face-on)
        slow: 0 for normal speed, 1 for slow motion
        split: Cross-validation split (1-4)
        youtube_id: YouTube ID or video identifier (default: same as video_id)
    """
    # Load database
    db_path = 'golfDB.pkl'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    df = pd.read_pickle(db_path)
    
    # Check if ID already exists
    if len(df.loc[df['id'] == video_id]) > 0:
        raise ValueError(f"Video ID {video_id} already exists in database")
    
    # Validate events
    if len(events) != 8:
        raise ValueError(f"Expected 8 events, got {len(events)}")
    
    # Check if video file exists
    video_path = f'videos_160/{video_id}.mp4'
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Convert events to numpy array
    events_array = np.array(events)
    
    # Set default bbox if not provided (full frame)
    if bbox is None:
        bbox = np.array([0.0, 0.0, 1.0, 1.0])  # [x, y, w, h] normalized
    else:
        bbox = np.array(bbox)
    
    # Use video_id as youtube_id if not provided
    if youtube_id is None:
        youtube_id = str(video_id)
    
    # Create new row
    new_row = {
        'id': video_id,
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
    print(f"\n✓ Added video {video_id} to database")
    print(f"  Video file: {video_path}")
    print(f"  Events: {events}")
    print(f"  Bbox: {bbox}")
    
    return video_id


if __name__ == '__main__':
    print("=" * 60)
    print("Add Existing Preprocessed Video to GolfDB Database")
    print("=" * 60)
    print()
    
    # Get video ID
    video_id_str = input("Enter video ID (e.g., 1402): ").strip()
    try:
        video_id = int(video_id_str)
    except ValueError:
        print(f"✗ Error: Invalid video ID: {video_id_str}")
        sys.exit(1)
    
    # Check if video file exists
    video_path = f'videos_160/{video_id}.mp4'
    if not os.path.exists(video_path):
        print(f"✗ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"✓ Found video: {video_path}")
    print()
    
    # Get events
    print("Enter 8 event frame numbers (space-separated):")
    print("  Format: Address Toe-up Mid-backswing Top Mid-downswing Impact Mid-follow-through Finish")
    print("  Example: 100 125 140 155 170 185 200 220")
    events_str = input("Events: ").strip()
    try:
        events = [int(x.strip()) for x in events_str.split()]
        if len(events) != 8:
            raise ValueError(f"Expected 8 events, got {len(events)}")
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
        if not bbox_str:
            bbox = None
            print("✓ Using full frame [0, 0, 1, 1]")
        else:
            bbox = [float(x.strip()) for x in bbox_str.split()]
            if len(bbox) != 4:
                raise ValueError(f"Expected 4 values (x y w h), got {len(bbox)}")
            if not all(0 <= x <= 1 for x in bbox):
                print("⚠ Warning: Bbox values should be normalized (0-1)")
            print(f"✓ Using bbox: {bbox}")
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    print()
    
    # Get youtube_id (optional)
    youtube_id = input(f"Enter YouTube ID/video identifier (default: {video_id}): ").strip()
    if not youtube_id:
        youtube_id = None
    else:
        print(f"✓ YouTube ID: {youtube_id}")
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
    
    # Confirm
    print("=" * 60)
    print("Summary:")
    print(f"  Video ID: {video_id}")
    print(f"  Video file: {video_path}")
    print(f"  Events: {events}")
    print(f"  Bbox: {bbox if bbox is not None else [0, 0, 1, 1]}")
    print(f"  YouTube ID: {youtube_id or video_id}")
    print(f"  Player: {player}")
    print(f"  Sex: {sex}")
    print(f"  Club: {club}")
    print(f"  View: {view}")
    print(f"  Slow motion: {slow}")
    print(f"  Split: {split}")
    print("=" * 60)
    print()
    
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)
    
    print()
    
    # Add to database
    try:
        add_existing_video_to_db(
            video_id=video_id,
            events=events,
            bbox=bbox,
            player=player,
            sex=sex,
            club=club,
            view=view,
            slow=slow,
            split=split,
            youtube_id=youtube_id
        )
        print(f"\n{'=' * 60}")
        print(f"✓ Successfully added video {video_id} to database")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

