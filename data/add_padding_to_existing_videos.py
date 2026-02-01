"""
Script to create new videos from existing videos_160 videos by:
1. Adding 100 duplicate frames of the first frame at the beginning
2. Adding 100 duplicate frames of the last frame at the end
3. Adjusting events in the database (shift by 100 frames)
4. Creating new video entries with ID = original_id + 1403
"""
import pandas as pd
import os
import cv2
import numpy as np

# Configuration
PADDING_FRAMES = 100  # Number of duplicate frames to add at start and end
ID_OFFSET = 1403  # Add this to original ID to get new ID
DIM = 160
INPUT_DIR = 'videos_160/'
OUTPUT_DIR = 'videos_160/'  # Same directory, different filenames

def add_padding_to_video(original_id, padding_frames=100, id_offset=1403):
    """
    Create a new video with padding frames at start and end.
    
    Args:
        original_id: Original video ID (e.g., 1207)
        padding_frames: Number of duplicate frames to add at start/end
        id_offset: Offset to add to original ID for new video ID
    
    Returns:
        Dictionary with new_id and adjusted events, or None if error
    """
    # Load database
    df = pd.read_pickle('golfDB.pkl')
    
    # Find original entry
    original_entry = df.loc[df['id'] == original_id]
    if len(original_entry) == 0:
        print(f'  ERROR: No database entry found for id {original_id}')
        return None
    
    original_row = original_entry.iloc[0]
    original_events = original_row['events'].copy()
    
    # Calculate new ID
    new_id = original_id + id_offset
    new_video_path = os.path.join(OUTPUT_DIR, f'{new_id}.mp4')
    
    # Check if new video already exists
    if os.path.isfile(new_video_path):
        print(f'  SKIP: New video {new_id}.mp4 already exists')
        return None
    
    # Open original video
    original_video_path = os.path.join(INPUT_DIR, f'{original_id}.mp4')
    if not os.path.isfile(original_video_path):
        print(f'  ERROR: Original video not found: {original_video_path}')
        return None
    
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f'  ERROR: Could not open video: {original_video_path}')
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f'  ERROR: Video has 0 frames: {original_video_path}')
        cap.release()
        return None
    
    print(f'  Processing: {original_id}.mp4 -> {new_id}.mp4')
    print(f'    Original: {total_frames} frames, {fps:.2f} fps')
    
    # Read first and last frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        print(f'  ERROR: Could not read first frame')
        cap.release()
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = cap.read()
    if not ret:
        print(f'  ERROR: Could not read last frame')
        cap.release()
        return None
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_video_path, fourcc, fps, (DIM, DIM))
    
    if not out.isOpened():
        print(f'  ERROR: Could not create output video: {new_video_path}')
        cap.release()
        return None
    
    frames_written = 0
    
    # Write padding frames at start (100x first frame)
    print(f'    Writing {padding_frames} duplicate frames at start...')
    for _ in range(padding_frames):
        out.write(first_frame)
        frames_written += 1
    
    # Write original video frames
    print(f'    Writing {total_frames} original frames...')
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
        frame_count += 1
    
    # Write padding frames at end (100x last frame)
    print(f'    Writing {padding_frames} duplicate frames at end...')
    for _ in range(padding_frames):
        out.write(last_frame)
        frames_written += 1
    
    cap.release()
    out.release()
    
    # Adjust events: shift by padding_frames (100)
    # Original events are relative to original video (starting at 0)
    # New events are relative to new video (starting at 100)
    adjusted_events = original_events + padding_frames
    
    print(f'    ✓ Created: {frames_written} total frames ({padding_frames} + {total_frames} + {padding_frames})')
    print(f'    Original events: {original_events}')
    print(f'    Adjusted events: {adjusted_events}')
    
    # Return new entry data
    return {
        'new_id': new_id,
        'original_id': original_id,
        'events': adjusted_events,
        'original_row': original_row
    }


def add_new_entries_to_database(new_entries):
    """
    Add new video entries to the database with adjusted IDs and events.
    """
    df = pd.read_pickle('golfDB.pkl')
    
    print(f'\n{'='*60}')
    print(f'Adding {len(new_entries)} new entries to database...')
    
    max_existing_id = df['id'].max()
    new_rows = []
    
    for entry in new_entries:
        if entry is None:
            continue
        
        original_row = entry['original_row']
        new_id = entry['new_id']
        adjusted_events = entry['events']
        
        # Create new row based on original, with new ID and adjusted events
        new_row = original_row.copy()
        new_row['id'] = new_id
        new_row['events'] = adjusted_events
        
        # Add to list
        new_rows.append(new_row)
        print(f'  Added entry: ID {new_id} (from {entry["original_id"]})')
    
    # Append new rows to dataframe
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save updated database
        df.to_pickle('golfDB.pkl')
        print(f'\n✓ Database updated: {len(new_rows)} new entries added')
        print(f'  Total entries: {len(df)}')
    else:
        print('  No new entries to add')


def regenerate_splits():
    """
    Regenerate train/val splits after database update.
    """
    df = pd.read_pickle('golfDB.pkl')
    
    print(f'\n{'='*60}')
    print('Regenerating train/val splits...')
    
    for i in range(1, 5):
        val_split = df.loc[df['split'] == i]
        val_split = val_split.reset_index()
        val_split = val_split.drop(columns=['index'])
        val_split.to_pickle("val_split_{:1d}.pkl".format(i))
        
        train_split = df.loc[df['split'] != i]
        train_split = train_split.reset_index()
        train_split = train_split.drop(columns=['index'])
        train_split.to_pickle("train_split_{:1d}.pkl".format(i))
    
    print(f'✓ Regenerated all train/val splits')


if __name__ == '__main__':
    import sys
    
    # Change to data directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load database
    df = pd.read_pickle('golfDB.pkl')
    print(f"Found {len(df)} videos in database")
    print(f"Will create new videos with ID offset: +{ID_OFFSET}")
    print(f"Padding: {PADDING_FRAMES} frames at start and end")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Process all videos
    new_entries = []
    total = len(df)
    successful = 0
    skipped = 0
    errors = 0
    
    for idx, row in df.iterrows():
        original_id = row['id']
        print(f"\n[{idx+1}/{total}] Processing ID {original_id}...")
        
        try:
            result = add_padding_to_video(original_id, PADDING_FRAMES, ID_OFFSET)
            if result:
                new_entries.append(result)
                successful += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            import traceback
            traceback.print_exc()
    
    # Add new entries to database
    if new_entries:
        add_new_entries_to_database(new_entries)
        regenerate_splits()
    else:
        print("\nNo new videos created, skipping database update")
    
    # Summary
    print(f"\n{'='*60}")
    print("✓ Processing complete!")
    print(f"  Successfully created: {successful} new videos")
    print(f"  Skipped: {skipped} videos")
    print(f"  Errors: {errors} videos")
    if successful > 0:
        print(f"  New videos saved to: {OUTPUT_DIR}")
        print(f"  Database updated with {successful} new entries")

