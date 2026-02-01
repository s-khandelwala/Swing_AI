import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np

df = pd.read_pickle('golfDB.pkl')
yt_video_dir = '../../database/videos/'


def preprocess_videos(anno_id, dim=160, padding_before=100, padding_after=100):
    """
    Extracts relevant frames from youtube videos with optional padding before/after swing.
    
    Args:
        anno_id: Annotation ID from database
        dim: Output video dimension (default: 160)
        padding_before: Number of frames to include before Address event (default: 100)
        padding_after: Number of frames to include after Finish event (default: 100)
    """
    # Reload database to get latest entries (in case it was updated)
    df = pd.read_pickle('golfDB.pkl')

    a = df.loc[df['id'] == anno_id]
    if len(a) == 0:
        raise ValueError(f"No entry found with id {anno_id}")
    
    # Use iloc[0] to get first row regardless of index
    bbox = a['bbox'].iloc[0]
    events = a['events'].iloc[0]  # Events in original video coordinates

    path = 'videos_{}/'.format(dim)

    if not os.path.isfile(os.path.join(path, "{}.mp4".format(anno_id))):
        print('Processing annotation id {}'.format(anno_id))
        cap = cv2.VideoCapture(os.path.join(yt_video_dir, '{}.mp4'.format(a['youtube_id'].iloc[0])))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {a['youtube_id'].iloc[0]}")
        
        total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range with padding
        # Events are in original video coordinates
        address_frame = events[0]  # First event (Address)
        finish_frame = events[-1]  # Last event (Finish)
        
        # Calculate start and end frames with padding
        start_frame = max(0, address_frame - padding_before)
        end_frame = min(total_frames_original - 1, finish_frame + padding_after)
        
        print(f'  Original video: {total_frames_original} frames')
        print(f'  Address at frame {address_frame}, Finish at frame {finish_frame}')
        print(f'  Including frames {start_frame} to {end_frame} (padding: {padding_before} before, {padding_after} after)')
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(path, "{}.mp4".format(anno_id)),
                              fourcc, fps, (dim, dim))
        x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[0])
        y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[2])
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[3])
        
        # Seek to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
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
        
        cap.release()
        out.release()
        print(f'  Processed {current_frame - start_frame} frames')
    else:
        print('Annotation id {} already completed for size {}'.format(anno_id, dim))


if __name__ == '__main__':
    path = 'videos_{}/'.format(160)
    if not os.path.exists(path):
        os.mkdir(path)
    preprocess_videos(df.id[1])
    # p = Pool(6)
    # p.map(preprocess_videos, df.id)
