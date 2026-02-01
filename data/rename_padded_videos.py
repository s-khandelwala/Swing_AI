"""
Script to rename padded videos (ID >= 1403) by decreasing ID by 2,
and update database entries based on the original video data.

This script:
1. Finds all videos with ID >= 1403 (padded videos)
2. Renames them to ID - 2 (e.g., 1403 -> 1401, 1404 -> 1402)
3. Updates database entries based on original video (original_id = new_id - 1401)
4. Adjusts events by subtracting 100 frames (removing padding offset)
"""

import pandas as pd
import os
import shutil
import numpy as np
import cv2

# Configuration
ID_OFFSET = 1403  # Original offset used when creating padded videos
PADDING_FRAMES = 100  # Number of padding frames added (to subtract from events)
VIDEO_DIR = 'videos_160/'
DECREASE_AMOUNT = 2  # Decrease ID by this amount (1403 -> 1401)

# Special case: video 2805 -> 2801, original_id = 1400, rotate 90 degrees
SPECIAL_CASE_OLD_ID = 2805
SPECIAL_CASE_NEW_ID = 2801
SPECIAL_CASE_ORIGINAL_ID = 1400
SPECIAL_CASE_ROTATE = 90  # degrees clockwise

def rotate_and_save_video(input_path, output_path, rotate_degrees=90):
    """
    Rotate a video by the specified degrees and save to output path.
    
    Args:
        input_path: Path to input video
        output_path: Path to save rotated video
        rotate_degrees: Rotation in degrees (90, 180, or 270)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine rotation code
    if rotate_degrees == 90:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotate_degrees == 180:
        rotate_code = cv2.ROTATE_180
    elif rotate_degrees == 270:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        raise ValueError(f"Unsupported rotation: {rotate_degrees}. Must be 90, 180, or 270")
    
    # Get frame size (will swap width/height for 90/270)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if rotate_degrees in [90, 270]:
        output_width, output_height = height, width
    else:
        output_width, output_height = width, height
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")
    
    print(f"    Rotating video by {rotate_degrees} degrees...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate frame
        rotated_frame = cv2.rotate(frame, rotate_code)
        out.write(rotated_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"    ✓ Rotated and saved {frame_count} frames")


def rename_padded_videos():
    """
    Rename all padded videos (ID >= 1403) by decreasing ID by 2,
    and update database entries based on original videos.
    """
    # Load database
    df = pd.read_pickle('golfDB.pkl')
    
    # Find all video files with ID >= 1403
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    video_ids = []
    for f in video_files:
        try:
            vid_id = int(f.replace('.mp4', ''))
            if vid_id >= ID_OFFSET:
                video_ids.append(vid_id)
        except ValueError:
            continue
    
    # Sort in ASCENDING order (lowest to highest) 
    # Process special case (2805) last
    video_ids = sorted(video_ids)
    
    if len(video_ids) == 0:
        print(f"No video files found with ID >= {ID_OFFSET}")
        return
    
    # Move special case (2805) to the end if it exists
    if SPECIAL_CASE_OLD_ID in video_ids:
        video_ids.remove(SPECIAL_CASE_OLD_ID)
        video_ids.append(SPECIAL_CASE_OLD_ID)
    
    print(f"Found {len(video_ids)} video files with ID >= {ID_OFFSET}")
    print(f"ID range: {video_ids[0]} to {video_ids[-1]}")
    print(f"Processing in ASCENDING order (lowest to highest), with special case last")
    print()
    
    # Track changes
    changes = []
    errors = []
    
    for old_id in video_ids:
        # Special case handling for video 2805
        if old_id == SPECIAL_CASE_OLD_ID:
            new_id = SPECIAL_CASE_NEW_ID
            original_id = SPECIAL_CASE_ORIGINAL_ID
            needs_rotation = True
            print(f"Processing SPECIAL CASE: {old_id} -> {new_id} (original: {original_id}, rotate: {SPECIAL_CASE_ROTATE}°)")
        else:
            new_id = old_id - DECREASE_AMOUNT
            # Calculate original video ID
            # If padded video was created as original_id + 1403,
            # then original_id = old_id - 1403
            original_id = old_id - ID_OFFSET
            needs_rotation = False
            print(f"Processing: {old_id} -> {new_id} (original: {original_id})")
        
        # Check if original video exists in database
        original_entry = df.loc[df['id'] == original_id]
        if len(original_entry) == 0:
            error_msg = f"  ERROR: Original video {original_id} not found in database"
            print(error_msg)
            errors.append((old_id, new_id, error_msg))
            continue
        
        original_row = original_entry.iloc[0]
        
        # Check if new_id already exists in database
        if len(df.loc[df['id'] == new_id]) > 0:
            error_msg = f"  ERROR: New ID {new_id} already exists in database"
            print(error_msg)
            errors.append((old_id, new_id, error_msg))
            continue
        
        # Check video files
        old_video_path = os.path.join(VIDEO_DIR, f'{old_id}.mp4')
        new_video_path = os.path.join(VIDEO_DIR, f'{new_id}.mp4')
        
        if not os.path.exists(old_video_path):
            error_msg = f"  ERROR: Video file not found: {old_video_path}"
            print(error_msg)
            errors.append((old_id, new_id, error_msg))
            continue
        
        # Skip overwrite check - allow overwriting if file exists
        
        # Get original events and adjust by adding padding_frames (100)
        # Padded videos have 100 frames added at the start, so events are shifted by +100
        # Original video events at [10, 25, 40] -> Padded video events at [110, 125, 140]
        original_events = original_row['events'].copy()
        new_events = original_events + PADDING_FRAMES
        
        print(f"  Original events: {original_events}")
        print(f"  Adjusted events: {new_events} (added {PADDING_FRAMES} for padding)")
        
        # Create new database entry based on original video
        new_row = original_row.copy()
        new_row['id'] = new_id
        new_row['events'] = new_events
        # Keep youtube_id from original
        new_row['youtube_id'] = original_row['youtube_id']
        
        # Record change
        changes.append({
            'old_id': old_id,
            'new_id': new_id,
            'original_id': original_id,
            'new_row': new_row,
            'old_video_path': old_video_path,
            'new_video_path': new_video_path,
            'needs_rotation': needs_rotation,
            'rotate_degrees': SPECIAL_CASE_ROTATE if needs_rotation else 0
        })
        
        print(f"  ✓ Prepared for rename")
    
    print()
    print("=" * 60)
    print(f"Summary: {len(changes)} videos ready to rename, {len(errors)} errors")
    print("=" * 60)
    
    if errors:
        print("\nErrors:")
        for old_id, new_id, error_msg in errors:
            print(f"  {old_id} -> {new_id}: {error_msg}")
    
    if len(changes) == 0:
        print("\nNo videos to rename. Exiting.")
        return
    
    # Confirm before proceeding
    print(f"\nReady to rename {len(changes)} videos:")
    print("  This will:")
    print("  1. Rename video files")
    print("  2. Update database entries")
    print("  3. Remove old database entries")
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    print()
    print("=" * 60)
    print("Renaming videos and updating database...")
    print("=" * 60)
    
    # Rename video files and update database
    df_new = df.copy()
    
    for change in changes:
        old_id = change['old_id']
        new_id = change['new_id']
        needs_rotation = change['needs_rotation']
        rotate_degrees = change['rotate_degrees']
        
        try:
            # Handle video file (rename or rotate and save)
            if needs_rotation:
                print(f"Rotating and saving {old_id}.mp4 -> {new_id}.mp4 (rotate {rotate_degrees}°)")
                rotate_and_save_video(change['old_video_path'], change['new_video_path'], rotate_degrees)
                # Delete original file after successful rotation
                if os.path.exists(change['old_video_path']):
                    os.remove(change['old_video_path'])
            else:
                print(f"Renaming {old_id}.mp4 -> {new_id}.mp4")
                shutil.move(change['old_video_path'], change['new_video_path'])
            
            # Remove old entry from database (if it exists)
            df_new = df_new[df_new['id'] != old_id]
            
            # Add new entry to database
            new_row_df = pd.DataFrame([change['new_row']])
            df_new = pd.concat([df_new, new_row_df], ignore_index=True)
            
            print(f"  ✓ Processed and updated database entry")
            
        except Exception as e:
            error_msg = f"  ERROR: {e}"
            print(error_msg)
            errors.append((old_id, new_id, error_msg))
            import traceback
            traceback.print_exc()
    
    # Save updated database
    print()
    print("Saving updated database...")
    df_new.to_pickle('golfDB.pkl')
    print("✓ Database saved")
    
    # Regenerate splits
    print()
    print("Regenerating train/val splits...")
    regenerate_splits(df_new)
    
    # Final summary
    print()
    print("=" * 60)
    print("✓ Renaming complete!")
    print(f"  Renamed: {len(changes)} videos")
    if errors:
        print(f"  Errors: {len(errors)} videos")
    print(f"  Database now has {len(df_new)} entries")
    print("=" * 60)


def regenerate_splits(df):
    """Regenerate train/val splits after database update."""
    for i in range(1, 5):
        val_split = df.loc[df['split'] == i]
        val_split = val_split.reset_index()
        val_split = val_split.drop(columns=['index'])
        val_split.to_pickle(f"val_split_{i}.pkl")
        
        train_split = df.loc[df['split'] != i]
        train_split = train_split.reset_index()
        train_split = train_split.drop(columns=['index'])
        train_split.to_pickle(f"train_split_{i}.pkl")
    
    print("✓ Regenerated all train/val splits")


if __name__ == '__main__':
    import sys
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("Rename Padded Videos Script")
    print("=" * 60)
    print()
    print(f"This script will rename all videos with ID >= {ID_OFFSET}")
    print(f"by decreasing their ID by {DECREASE_AMOUNT} (e.g., 1403 -> 1401)")
    print(f"and update database entries based on original videos.")
    print()
    
    rename_padded_videos()

