"""
Edit event frames (phases) for existing videos in the golfDB.pkl database.
"""
import pandas as pd
import numpy as np
import os
import sys
import cv2
import subprocess
import json

# Event names for reference
event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

# Configuration for padded videos
ID_OFFSET = 1403  # Padded videos have ID = original_id + 1403
PADDING_FRAMES = 100  # Number of frames added at start of padded videos

def get_video_rotation(video_path):
    """Detect video rotation from metadata using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Check for rotation tag first (most reliable)
                rotation = stream.get('tags', {}).get('rotate')
                if rotation:
                    rot_int = int(rotation)
                    rot_int = rot_int % 360
                    if rot_int == 90:
                        return 90
                    elif rot_int == 180:
                        return 180
                    elif rot_int == 270:
                        return 270
                
                # Check displaymatrix side data
                side_data_list = stream.get('side_data_list', [])
                for side_data in side_data_list:
                    if side_data.get('side_data_type') == 'Display Matrix':
                        matrix_str = side_data.get('displaymatrix', '')
                        if '-65536' in matrix_str or '65536' in matrix_str:
                            return 90
    except Exception:
        pass  # If ffprobe fails, assume no rotation
    
    return 0

def show_video_frames(video_path, events, rotate=0):
    """Display video frames at event positions with optional rotation"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    for i, (event_name, frame_num) in enumerate(event_names.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, events[i])
        ret, img = cap.read()
        if ret:
            # Apply rotation if needed
            if rotate == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rotate == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Add frame number text
            cv2.putText(img, f'{event_name}: Frame {events[i]}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            window_name = f'{event_name} (Frame {events[i]})'
            cv2.imshow(window_name, img)
            print(f"Showing {event_name} at frame {events[i]} (press any key to continue)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Could not read frame {events[i]} for {event_name}")
    
    cap.release()

def load_database():
    """Load the golfDB.pkl database"""
    db_path = 'golfDB.pkl'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    return pd.read_pickle(db_path)

def save_database(df):
    """Save the golfDB.pkl database"""
    db_path = 'golfDB.pkl'
    df.to_pickle(db_path)
    print(f"Database saved to {db_path}")

def show_video_info(df, video_id, show_video=False, video_dir='videos_160/'):
    """Display current video information"""
    row = df.loc[df['id'] == video_id]
    if len(row) == 0:
        print(f"Video ID {video_id} not found in database")
        print(f"Database contains IDs {df['id'].min()} to {df['id'].max()} ({len(df)} entries)")
        
        # Check if it might be a padded video
        if video_id >= ID_OFFSET:
            original_id = video_id - ID_OFFSET
            original_exists = len(df.loc[df['id'] == original_id]) > 0
            if original_exists:
                print(f"Note: This appears to be a padded video (ID {video_id} = original ID {original_id} + {ID_OFFSET})")
                print(f"      Original video (ID {original_id}) exists in database.")
            else:
                print(f"Note: This appears to be a padded video, but original ID {original_id} not found.")
        else:
            # Check if video file exists
            video_path = os.path.join(video_dir, f'{video_id}.mp4')
            if os.path.exists(video_path):
                print(f"Note: Video file {video_path} exists but is not in the database.")
                print(f"      You may need to add it to the database first.")
        
        return None
    
    row = row.iloc[0]
    print(f"\n{'='*60}")
    print(f"Video ID: {video_id}")
    print(f"{'='*60}")
    print(f"YouTube ID: {row['youtube_id']}")
    print(f"Player: {row['player']}")
    print(f"Sex: {row['sex']}")
    print(f"Club: {row['club']}")
    print(f"View: {row['view']}")
    print(f"\nCurrent Event Frames:")
    events = row['events']
    for i, event_name in event_names.items():
        print(f"  {event_name:35s}: Frame {events[i]}")
    print(f"{'='*60}\n")
    
    # Show video frames if requested
    if show_video:
        video_path = os.path.join(video_dir, f'{video_id}.mp4')
        if os.path.exists(video_path):
            print("Detecting video rotation...")
            rotation = get_video_rotation(video_path)
            if rotation != 0:
                print(f"Video has rotation metadata: {rotation} degrees")
            print("\nDisplaying event frames (press any key to advance)...")
            show_video_frames(video_path, events, rotate=rotation)
        else:
            print(f"Video file not found: {video_path}")
            print("Skipping video display.")
    
    return row

def update_padded_video_events(df, original_id, new_events):
    """Update events for the corresponding padded video (original_id + ID_OFFSET)"""
    padded_id = original_id + ID_OFFSET
    
    # Check if padded video exists
    padded_row = df.loc[df['id'] == padded_id]
    if len(padded_row) == 0:
        return False  # No padded video exists
    
    # Calculate new events for padded video (add PADDING_FRAMES to each event)
    padded_events = new_events + PADDING_FRAMES
    
    # Update the database
    df.loc[df['id'] == padded_id, 'events'] = pd.Series([padded_events], index=df.loc[df['id'] == padded_id].index)
    
    return True

def edit_events_interactive(df, video_id):
    """Interactively edit event frames for a video"""
    row = show_video_info(df, video_id)
    if row is None:
        return False
    
    events = row['events'].copy()
    print("Enter new frame numbers for each event (press Enter to keep current value):")
    print()
    
    new_events = []
    for i, event_name in event_names.items():
        current_frame = events[i]
        try:
            user_input = input(f"{event_name:35s} [current: {current_frame}]: ").strip()
            if user_input == '':
                new_events.append(current_frame)
            else:
                new_frame = int(user_input)
                if new_frame < 0:
                    print(f"  Warning: Frame number must be >= 0, keeping current value {current_frame}")
                    new_events.append(current_frame)
                else:
                    new_events.append(new_frame)
        except ValueError:
            print(f"  Warning: Invalid input, keeping current value {current_frame}")
            new_events.append(current_frame)
        except KeyboardInterrupt:
            print("\n\nEdit cancelled.")
            return False
    
    # Verify events are in ascending order
    new_events = np.array(new_events)
    if not np.all(new_events[:-1] <= new_events[1:]):
        print("\nWarning: Event frames are not in ascending order!")
        print("Event frames should increase from Address to Finish.")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Edit cancelled.")
            return False
    
    # Update the database for original video
    df.loc[df['id'] == video_id, 'events'] = pd.Series([new_events], index=df.loc[df['id'] == video_id].index)
    
    print("\nUpdated Event Frames:")
    for i, event_name in event_names.items():
        print(f"  {event_name:35s}: Frame {new_events[i]}")
    
    # Check if this is an original video (not a padded one) and update padded video if it exists
    if video_id < ID_OFFSET:  # Likely an original video (padded videos have ID >= 1403)
        padded_id = video_id + ID_OFFSET
        padded_exists = len(df.loc[df['id'] == padded_id]) > 0
        if padded_exists:
            if update_padded_video_events(df, video_id, new_events):
                padded_events = new_events + PADDING_FRAMES
                print(f"\nAlso updated corresponding padded video (ID {padded_id}):")
                for i, event_name in event_names.items():
                    print(f"  {event_name:35s}: Frame {padded_events[i]}")
    
    return True

def edit_events_from_list(df, video_id, new_events_list, update_padded=True):
    """Edit events using a list of 8 frame numbers"""
    if len(new_events_list) != 8:
        raise ValueError(f"Expected 8 event frames, got {len(new_events_list)}")
    
    row = df.loc[df['id'] == video_id]
    if len(row) == 0:
        raise ValueError(f"Video ID {video_id} not found in database")
    
    new_events = np.array(new_events_list, dtype=int)
    
    # Verify events are in ascending order
    if not np.all(new_events[:-1] <= new_events[1:]):
        raise ValueError("Event frames must be in ascending order (Address to Finish)")
    
    # Update the database for original video
    df.loc[df['id'] == video_id, 'events'] = pd.Series([new_events], index=df.loc[df['id'] == video_id].index)
    
    # Check if this is an original video (not a padded one) and update padded video if it exists
    if update_padded and video_id < ID_OFFSET:  # Likely an original video
        padded_id = video_id + ID_OFFSET
        padded_exists = len(df.loc[df['id'] == padded_id]) > 0
        if padded_exists:
            update_padded_video_events(df, video_id, new_events)
            print(f"Also updated corresponding padded video (ID {padded_id})")
    
    return True

def list_videos(df, limit=20):
    """List videos in the database"""
    print(f"\n{'='*60}")
    print(f"Videos in database (showing first {limit}):")
    print(f"{'='*60}")
    print(f"{'ID':<8} {'YouTube ID':<20} {'Player':<15} {'Events'}")
    print(f"{'-'*60}")
    
    for idx, row in df.head(limit).iterrows():
        events_str = f"{row['events'][0]}-{row['events'][-1]}"
        print(f"{row['id']:<8} {str(row['youtube_id']):<20} {str(row['player']):<15} {events_str}")
    
    if len(df) > limit:
        print(f"\n... and {len(df) - limit} more videos")
    print(f"{'='*60}\n")

def main():
    """Main function for interactive editing"""
    print("Golf Video Event Editor")
    print("=" * 60)
    
    # Load database
    try:
        df = load_database()
        print(f"Loaded database with {len(df)} videos")
    except Exception as e:
        print(f"Error loading database: {e}")
        return
    
    while True:
        print("\nOptions:")
        print("  1. List videos")
        print("  2. Edit events for a video")
        print("  3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            list_videos(df)
        
        elif choice == '2':
            try:
                video_id = int(input("Enter video ID to edit: ").strip())
                
                # Ask if user wants to view video first
                view_video = input("View video frames at event positions first? (y/n): ").strip().lower() == 'y'
                view_padded = False
                if view_video and video_id < ID_OFFSET:
                    # If viewing original video, offer to also view padded version
                    padded_id = video_id + ID_OFFSET
                    if len(df.loc[df['id'] == padded_id]) > 0:
                        view_padded = input(f"Also view corresponding padded video (ID {padded_id})? (y/n): ").strip().lower() == 'y'
                
                # Show info and optionally display video
                row = show_video_info(df, video_id, show_video=view_video)
                if row is None:
                    continue
                
                # Show padded video if requested
                if view_padded:
                    padded_id = video_id + ID_OFFSET
                    padded_row = df.loc[df['id'] == padded_id]
                    if len(padded_row) > 0:
                        print(f"\n{'='*60}")
                        print(f"Showing padded video (ID {padded_id})")
                        print(f"{'='*60}")
                        show_video_info(df, padded_id, show_video=True)
                
                if edit_events_interactive(df, video_id):
                    save_choice = input("\nSave changes? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        save_database(df)
                        print("Changes saved!")
                    else:
                        print("Changes discarded. Reloading database...")
                        df = load_database()
            except ValueError:
                print("Invalid video ID. Please enter a number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    # Check if running as script with arguments
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='Edit event frames for videos in golfDB.pkl')
        parser.add_argument('--video-id', type=int, help='Video ID to edit')
        parser.add_argument('--events', nargs=8, type=int, metavar=('ADDRESS', 'TOE_UP', 'MID_BACK', 'TOP', 'MID_DOWN', 'IMPACT', 'MID_FOLLOW', 'FINISH'),
                           help='8 event frame numbers (Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish)')
        parser.add_argument('--list', action='store_true', help='List all videos')
        parser.add_argument('--show', type=int, help='Show info for a specific video ID')
        parser.add_argument('--show-video', action='store_true', help='Display video frames at event positions (use with --show)')
        
        args = parser.parse_args()
        
        df = load_database()
        
        if args.list:
            list_videos(df, limit=1000)  # Show all
        
        elif args.show:
            show_video_info(df, args.show, show_video=args.show_video)
        
        elif args.video_id and args.events:
            try:
                if edit_events_from_list(df, args.video_id, args.events):
                    save_database(df)
                    print(f"Successfully updated events for video ID {args.video_id}")
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        
        else:
            parser.print_help()
    else:
        # Run interactive mode
        main()

