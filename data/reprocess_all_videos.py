"""
Script to reprocess all videos with padding and update the database.
This will:
1. Reprocess all videos with padding_before and padding_after
2. Update events in the database to be relative to preprocessed video coordinates
3. Regenerate train/val splits
"""
import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np
# from preprocess_videos import preprocess_videos  # Not needed, we have our own function

# Configuration
PADDING_BEFORE = 100
PADDING_AFTER = 100
DIM = 160
NUM_WORKERS = 6  # Set to 1 for sequential processing, or higher for parallel
YT_VIDEO_DIR = 'videos_160/'  # Path to original YouTube videos (or preprocessed if adding padding isn't needed)
USE_EXISTING_PREPROCESSED = True  # If True, use existing videos_160 and just adjust events (no padding added)

def preprocess_with_event_adjustment(anno_id, dim=160, padding_before=100, padding_after=100):
    """
    Preprocess a video with padding and return adjusted events.
    Events are adjusted to be relative to the preprocessed video start.
    """
    # Reload database to get latest entries
    df = pd.read_pickle('golfDB.pkl')
    
    a = df.loc[df['id'] == anno_id]
    if len(a) == 0:
        raise ValueError(f"No entry found with id {anno_id}")
    
    bbox = a['bbox'].iloc[0]
    events_original = a['events'].iloc[0].copy()  # Events in original video coordinates
    
    path = 'videos_{}/'.format(dim)
    
    # Check if video already exists
    video_path = os.path.join(path, "{}.mp4".format(anno_id))
    # SAFETY: Never delete existing files - if video exists, skip reprocessing
    if os.path.isfile(video_path):
        if USE_EXISTING_PREPROCESSED:
            print(f'  Using existing video: {video_path}')
        else:
            print(f'  Video {anno_id} already exists, skipping reprocessing (file deletion disabled for safety)')
            # Return None to skip this video since we can't overwrite it
            return None
    
    print(f'Processing annotation id {anno_id}')
    youtube_id = a['youtube_id'].iloc[0]
    
    if USE_EXISTING_PREPROCESSED:
        # Use existing preprocessed video (named by annotation ID, not YouTube ID)
        source_video_path = os.path.join(YT_VIDEO_DIR, '{}.mp4'.format(anno_id))
        # Note: Can't add padding from preprocessed videos, but we can adjust events
        print(f'  Using existing preprocessed video: {source_video_path}')
        if not os.path.isfile(source_video_path):
            print(f'  SKIP: Preprocessed video not found: {source_video_path}')
            return None
        # For existing preprocessed videos, events should already be relative to video start
        # So we just need to verify/adjust them
        cap = cv2.VideoCapture(source_video_path)
        total_frames_preprocessed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Events in database are in original coordinates, but preprocessed video starts at Address
        # So Address should be at frame 0 in preprocessed video
        # Adjust events: subtract Address frame to make it frame 0
        events_adjusted = events_original - events_original[0]
        print(f'  Using existing video: {total_frames_preprocessed} frames')
        print(f'  Original events: Address={events_original[0]}, Finish={events_original[-1]}')
        print(f'  Adjusted events: Address={events_adjusted[0]}, Finish={events_adjusted[-1]}')
        # Don't reprocess, just return adjusted events
        cap.release()
        return {
            'id': anno_id,
            'events': events_adjusted
        }
    else:
        # Use original YouTube videos (named by YouTube ID)
        original_video_path = os.path.join(YT_VIDEO_DIR, '{}.mp4'.format(youtube_id))
        
        # Check if original video exists
        if not os.path.isfile(original_video_path):
            print(f'  SKIP: Original video not found: {original_video_path}')
            print(f'        (Need original YouTube videos to add padding)')
            return None
        
        cap = cv2.VideoCapture(original_video_path)
    
    if not cap.isOpened():
        print(f'  ERROR: Could not open video: {youtube_id}')
        return None
    
        total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range with padding
        address_frame = events_original[0]  # First event (Address)
        finish_frame = events_original[-1]  # Last event (Finish)
        
        # Calculate start and end frames with padding
        start_frame = max(0, address_frame - padding_before)
        end_frame = min(total_frames_original - 1, finish_frame + padding_after)
        
        # Adjust events to be relative to preprocessed video start
        # In preprocessed video, Address will be at (address_frame - start_frame)
        events_adjusted = events_original - start_frame
    
        print(f'  Original: Address={address_frame}, Finish={finish_frame}')
        print(f'  Preprocessed: Address={events_adjusted[0]}, Finish={events_adjusted[-1]}')
        print(f'  Frame range: {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)')
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (dim, dim))
        x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[0])
        y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[2])
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[3])
        
        # Seek to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        frames_processed = 0
        
        while current_frame <= end_frame:
            ret, image = cap.read()
            if not ret:
                break
            
            # Process frame: crop, resize, and pad
            crop_img = image[y:y + h, x:x + w]
            crop_size = crop_img.shape[:2]
            ratio = dim / max(crop_size)
            new_size = tuple([int(x*ratio) for x in crop_size])
            resized = cv2.resize(crop_img, (new_size[1], new_size[0]))
            delta_w = dim - new_size[1]
            delta_h = dim - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406*255, 0.456*255, 0.485*255])  # ImageNet means (BGR)
            out.write(b_img)
            current_frame += 1
            frames_processed += 1
        
        cap.release()
        out.release()
        
        print(f'  ✓ Processed {frames_processed} frames')
    
    # Return adjusted events for database update
    return {
        'id': anno_id,
        'events': events_adjusted
    }


def update_database_events(updates):
    """
    Update events in the database with adjusted values.
    """
    df = pd.read_pickle('golfDB.pkl')
    
    for update in updates:
        if update is None:
            continue
        anno_id = update['id']
        events_adjusted = update['events']
        
        # Find the row and update events
        idx = df.index[df['id'] == anno_id].tolist()
        if len(idx) > 0:
            df.at[idx[0], 'events'] = events_adjusted
            print(f'  Updated events for id {anno_id}')
        else:
            print(f'  WARNING: Could not find id {anno_id} in database')
    
    # Save updated database
    df.to_pickle('golfDB.pkl')
    print(f'\n✓ Database updated and saved')


def regenerate_splits():
    """
    Regenerate train/val splits after database update.
    """
    # Regenerate splits from the updated .pkl
    df = pd.read_pickle('golfDB.pkl')
    
    for i in range(1, 5):
        val_split = df.loc[df['split'] == i]
        val_split = val_split.reset_index()
        val_split = val_split.drop(columns=['index'])
        val_split.to_pickle("val_split_{:1d}.pkl".format(i))
        
        train_split = df.loc[df['split'] != i]
        train_split = train_split.reset_index()
        train_split = train_split.drop(columns=['index'])
        train_split.to_pickle("train_split_{:1d}.pkl".format(i))
    
    print(f'\n✓ Regenerated all train/val splits')


if __name__ == '__main__':
    import sys
    
    # Change to data directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if video directory exists
    if not os.path.exists(YT_VIDEO_DIR):
        print(f"ERROR: Video directory not found: {YT_VIDEO_DIR}")
        if USE_EXISTING_PREPROCESSED:
            print(f"       Looking for preprocessed videos in videos_160/")
        else:
            print(f"       You need the original YouTube videos to add padding.")
            print(f"       The directory should contain videos like 'f1BWA5F87Jc.mp4'")
        sys.exit(1)
    
    if USE_EXISTING_PREPROCESSED:
        print(f"NOTE: Using existing preprocessed videos from {YT_VIDEO_DIR}")
        print(f"      Events will be adjusted but NO padding will be added.")
        print(f"      (To add padding, set USE_EXISTING_PREPROCESSED=False and provide original videos)")
    else:
        print(f"NOTE: Using original YouTube videos from {YT_VIDEO_DIR}")
        print(f"      Videos will be reprocessed with padding.")
    
    # Load database
    df = pd.read_pickle('golfDB.pkl')
    print(f"Found {len(df)} videos to process")
    print(f"Padding: {PADDING_BEFORE} frames before, {PADDING_AFTER} frames after")
    print(f"Output dimension: {DIM}x{DIM}")
    print(f"Original videos from: {YT_VIDEO_DIR}")
    print()
    
    # Create output directory
    path = 'videos_{}/'.format(DIM)
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Process videos
    if NUM_WORKERS > 1:
        print(f"Processing with {NUM_WORKERS} workers...")
        # For parallel processing, we need to handle event updates differently
        # For now, let's do sequential to ensure database updates are correct
        print("Note: Using sequential processing to ensure database consistency")
        NUM_WORKERS = 1
    
    if NUM_WORKERS == 1:
        # Sequential processing
        updates = []
        skipped = []
        errors = []
        total = len(df)
        for idx, row in df.iterrows():
            anno_id = row['id']
            print(f"\n[{idx+1}/{total}] Processing id {anno_id}...")
            try:
                result = preprocess_with_event_adjustment(anno_id, dim=DIM, 
                                                          padding_before=PADDING_BEFORE, 
                                                          padding_after=PADDING_AFTER)
                if result:
                    updates.append(result)
                else:
                    skipped.append(anno_id)
            except Exception as e:
                print(f"  ERROR processing {anno_id}: {e}")
                errors.append((anno_id, str(e)))
                import traceback
                traceback.print_exc()
        
        # Update database with adjusted events
        print(f"\n{'='*60}")
        print("Updating database with adjusted events...")
        if updates:
            update_database_events(updates)
        else:
            print("  No videos processed, skipping database update")
        
        # Regenerate splits
        print(f"\n{'='*60}")
        print("Regenerating train/val splits...")
        if updates:
            regenerate_splits()
        else:
            print("  No videos processed, skipping split regeneration")
        
        print(f"\n{'='*60}")
        print("✓ Processing complete!")
        print(f"  Successfully processed: {len(updates)} videos")
        print(f"  Skipped (no source video): {len(skipped)} videos")
        print(f"  Errors: {len(errors)} videos")
        if updates:
            print(f"  Database: golfDB.pkl updated")
            print(f"  Splits: train_split_*.pkl and val_split_*.pkl regenerated")
        if skipped:
            print(f"\n  Note: {len(skipped)} videos were skipped because original")
            print(f"        YouTube videos were not found in {YT_VIDEO_DIR}")
            print(f"        To process these, download the original videos first.")
    else:
        # Parallel processing (not implemented yet due to database update complexity)
        print("Parallel processing not yet implemented. Use NUM_WORKERS=1")

