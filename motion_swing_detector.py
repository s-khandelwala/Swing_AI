"""
Pose-Based Golf Swing Boundary Detection

Uses MediaPipe pose estimation to automatically detect when a golf swing starts and ends
by tracking the golfer's body keypoints (wrists, shoulders, hips).
"""

import cv2
import numpy as np
import os
import os.path as osp
from scipy import ndimage
from typing import Tuple, Optional

# Try to import MediaPipe with support for both old and new APIs
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_USE_NEW_API = False
mp = None
mp_pose = None

try:
    import mediapipe as mp
    
    # Check for new API (MediaPipe Tasks API - 0.10.30+)
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as mp_base_options
        # Try to actually create an object to verify it works
        # For MediaPipe 0.10.31+, BaseOptions can be created without model_asset_path
        # (it will use bundled model or require one to be specified later)
        test_base = mp_base_options.BaseOptions()
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_USE_NEW_API = True
        print("✅ MediaPipe detected (new Tasks API)")
    except (ImportError, AttributeError, Exception) as e:
        # Fall back to old API (mediapipe.solutions - <0.10.30)
        # Try to actually access solutions.pose to verify it works
        try:
            test_pose = mp.solutions.pose
            # Try to create a Pose instance to verify it works
            test_instance = test_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Use simplest for test
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_USE_NEW_API = False
            print("✅ MediaPipe detected (legacy solutions API)")
            # Clean up test instance
            test_instance.close()
        except (AttributeError, Exception) as e2:
            MEDIAPIPE_AVAILABLE = False
            print("⚠️  MediaPipe installed but neither old nor new API found")
            print(f"   New API error: {type(e).__name__}: {e}")
            print(f"   Legacy API error: {type(e2).__name__}: {e2}")
            print("   Try: pip install 'mediapipe<0.10.30' for legacy API")
            print("   Or: pip install --upgrade mediapipe for new API")
            print("   Or check MediaPipe version: pip show mediapipe")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - install with: pip install mediapipe")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️  MediaPipe error: {e}")


class PoseBasedSwingDetector:
    """
    Pose-based swing detection using MediaPipe.
    Tracks golfer's body keypoints to detect swing start/end.
    More reliable than motion-based detection as it tracks golfer intent, not background.
    """
    
    def __init__(self, 
                 min_confidence=0.5,
                 smoothing_window=5,
                 start_motion_threshold=0.15,  # Relative to peak (15% of way from baseline to peak)
                 end_motion_threshold=0.10,    # Relative to peak (10% of way from baseline to peak)
                 min_sustained_frames=8):      # Frames for sustained motion (filters waggles)
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window
        self.start_motion_threshold = start_motion_threshold
        self.end_motion_threshold = end_motion_threshold
        self.min_sustained_frames = min_sustained_frames
        
        if MEDIAPIPE_AVAILABLE:
            if MEDIAPIPE_USE_NEW_API:
                # New MediaPipe Tasks API (0.10.30+)
                try:
                    from mediapipe.tasks.python import vision
                    from mediapipe.tasks.python.core import base_options as mp_base_options
                    import os
                    import urllib.request
                    import tempfile
                    
                    # For MediaPipe 0.10.31+, we need to download a model file
                    # Download the pose landmarker model if not already cached
                    model_dir = os.path.join(tempfile.gettempdir(), 'mediapipe_models')
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')
                    
                    if not os.path.exists(model_path):
                        print("   📥 Downloading MediaPipe pose landmarker model (this is a one-time download)...")
                        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                        try:
                            urllib.request.urlretrieve(model_url, model_path)
                            print("   ✅ Model downloaded successfully")
                        except Exception as download_error:
                            print(f"   ⚠️  Failed to download model: {download_error}")
                            print("   💡 Pose detection will be disabled.")
                            self.pose = None
                            self.use_new_api = False
                            return
                    
                    # Create BaseOptions with the model file
                    base_options = mp_base_options.BaseOptions(model_asset_path=model_path)
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        output_segmentation_masks=False,
                        min_pose_detection_confidence=min_confidence,
                        min_pose_presence_confidence=min_confidence,
                        min_tracking_confidence=min_confidence
                    )
                    self.pose = vision.PoseLandmarker.create_from_options(options)
                    self.mp_pose = None  # Not used in new API
                    self.use_new_api = True
                except Exception as e:
                    print(f"⚠️  Failed to initialize new MediaPipe API: {e}")
                    self.pose = None
                    self.use_new_api = False
            else:
                # Legacy MediaPipe Solutions API (<0.10.30)
                try:
                    self.mp_pose = mp.solutions.pose
                    self.pose = self.mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,  # 0=fast, 1=balanced, 2=accurate
                        smooth_landmarks=True,
                        min_detection_confidence=min_confidence,
                        min_tracking_confidence=min_confidence
                    )
                    self.use_new_api = False
                except Exception as e:
                    print(f"⚠️  Failed to initialize legacy MediaPipe API: {e}")
                    self.pose = None
                    self.use_new_api = False
        else:
            self.pose = None
            self.use_new_api = False
    
    def _extract_keypoints(self, frame):
        """Extract key body points from frame."""
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Handle both old and new MediaPipe APIs
        landmarks = None
        if hasattr(self, 'use_new_api') and self.use_new_api:
            # New MediaPipe Tasks API
            try:
                from mediapipe import ImageFormat
                mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.pose.detect(mp_image)
                
                if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                    return None
                
                landmarks = detection_result.pose_landmarks[0]  # First person detected
            except Exception as e:
                return None
        else:
            # Legacy MediaPipe Solutions API
            try:
                results = self.pose.process(rgb_frame)
                
                if not results.pose_landmarks:
                    return None
                
                landmarks = results.pose_landmarks.landmark
            except Exception as e:
                return None
        
        if landmarks is None:
            return None
        
        # Extract key points for swing detection
        # MediaPipe pose landmark indices (same for both APIs):
        # https://google.github.io/mediapipe/solutions/pose.html
        # For right-handed golfer: left_wrist is lead hand, right_wrist is trail hand
        
        try:
            # Both APIs use the same landmark structure
            keypoints = {
                'left_wrist': (landmarks[15].x, landmarks[15].y),      # Lead hand (for right-handed golfer)
                'right_wrist': (landmarks[16].x, landmarks[16].y),    # Trail hand
                'left_shoulder': (landmarks[11].x, landmarks[11].y),
                'right_shoulder': (landmarks[12].x, landmarks[12].y),
                'left_hip': (landmarks[23].x, landmarks[23].y),
                'right_hip': (landmarks[24].x, landmarks[24].y),
            }
            
            # Calculate shoulder center and hip center
            keypoints['shoulder_center'] = (
                (keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) / 2,
                (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2
            )
            keypoints['hip_center'] = (
                (keypoints['left_hip'][0] + keypoints['right_hip'][0]) / 2,
                (keypoints['left_hip'][1] + keypoints['right_hip'][1]) / 2
            )
            
            return keypoints
        except (IndexError, AttributeError) as e:
            return None
    
    def _compute_motion_score(self, keypoints_prev, keypoints_curr):
        """
        Compute combined motion score from keypoints.
        
        Formula:
        motion = 0.5 * lead_wrist_velocity + 
                 0.3 * trail_wrist_velocity + 
                 0.2 * shoulder_rotation_speed
        """
        if keypoints_prev is None or keypoints_curr is None:
            return 0.0
        
        # Lead wrist velocity (primary signal)
        lead_wrist_dist = np.sqrt(
            (keypoints_curr['left_wrist'][0] - keypoints_prev['left_wrist'][0])**2 +
            (keypoints_curr['left_wrist'][1] - keypoints_prev['left_wrist'][1])**2
        )
        
        # Trail wrist velocity
        trail_wrist_dist = np.sqrt(
            (keypoints_curr['right_wrist'][0] - keypoints_prev['right_wrist'][0])**2 +
            (keypoints_curr['right_wrist'][1] - keypoints_prev['right_wrist'][1])**2
        )
        
        # Shoulder rotation (distance between shoulder centers)
        shoulder_rotation = np.sqrt(
            (keypoints_curr['shoulder_center'][0] - keypoints_prev['shoulder_center'][0])**2 +
            (keypoints_curr['shoulder_center'][1] - keypoints_prev['shoulder_center'][1])**2
        )
        
        # Combined motion score
        motion_score = (
            0.5 * lead_wrist_dist +
            0.3 * trail_wrist_dist +
            0.2 * shoulder_rotation
        )
        
        return motion_score
    
    def detect_swing_boundaries(self, video_path, fps, subsample_rate=1):
        """
        Detect swing boundaries using pose tracking.
        
        Args:
            video_path: Path to video file
            fps: Video frame rate
            subsample_rate: Process every Nth frame
            
        Returns:
            (start_frame, end_frame, confidence) or None if detection fails
        """
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        motion_scores = []
        frame_indices = []
        keypoints_prev = None
        frame_count = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % subsample_rate == 0:
                keypoints = self._extract_keypoints(frame)
                
                if keypoints is not None:
                    if keypoints_prev is not None:
                        motion_score = self._compute_motion_score(keypoints_prev, keypoints)
                        motion_scores.append(motion_score)
                        frame_indices.append(frame_count)
                    
                    keypoints_prev = keypoints
                # If no pose detected, skip this frame (handles occlusions)
            
            frame_count += 1
        
        cap.release()
        
        if len(motion_scores) < self.min_sustained_frames:
            return None
        
        # Smooth motion scores
        motion_scores = np.array(motion_scores)
        if len(motion_scores) > self.smoothing_window:
            from scipy.ndimage import uniform_filter1d
            motion_scores = uniform_filter1d(motion_scores, size=self.smoothing_window)
        
        # Find address period (stable, low motion)
        # Look for first 0.5 seconds of low motion
        address_window = max(1, int(0.5 * fps / subsample_rate))
        address_end_idx = None
        
        for i in range(len(motion_scores) - address_window):
            window = motion_scores[i:i+address_window]
            if np.mean(window) < np.percentile(motion_scores, 20):
                address_end_idx = i + address_window
                break
        
        if address_end_idx is None:
            address_end_idx = min(address_window, len(motion_scores) - 1)
        
        # Find baseline and peak
        baseline = np.median(motion_scores[:address_end_idx]) if address_end_idx > 0 else np.median(motion_scores)
        peak = np.max(motion_scores)
        motion_range = peak - baseline
        
        if motion_range < 1e-6:
            return None  # No significant motion
        
        # Adaptive thresholds (relative, not absolute - works for slow motion!)
        start_threshold = baseline + self.start_motion_threshold * motion_range
        end_threshold = baseline + self.end_motion_threshold * motion_range
        
        # Detect SWING START: find where motion starts increasing toward the peak
        # Simple approach: work backwards from peak, find the local minimum or inflection point
        start_idx = None
        
        # Find the peak first
        peak_idx = np.argmax(motion_scores)
        
        # Smooth motion for better analysis
        try:
            from scipy.ndimage import uniform_filter1d
            motion_smooth = uniform_filter1d(motion_scores, size=5)
        except:
            kernel = np.ones(5) / 5
            motion_smooth = np.convolve(motion_scores, kernel, mode='same')
        
        # Simple approach: find the minimum point before the peak, then find where motion starts rising from there
        search_start = max(address_end_idx, peak_idx - 60)  # Look back up to 60 frames
        search_end = peak_idx - 5  # Don't go too close to peak
        
        if search_end > search_start:
            # Find local minimum in the region before the peak
            search_region = motion_smooth[search_start:search_end]
            if len(search_region) > 0:
                local_min_idx = np.argmin(search_region) + search_start
                
                # Now work forward from the local minimum to find where motion starts consistently increasing
                # This is where the swing actually starts
                for i in range(local_min_idx, min(peak_idx - 3, len(motion_scores) - 1)):
                    # Look ahead to see if motion is increasing
                    look_ahead = min(10, peak_idx - i)
                    if look_ahead < 5:
                        break
                    
                    # Check if motion increases over the next few frames
                    current_motion = motion_smooth[i]
                    future_motion = motion_smooth[i+1:min(i+look_ahead+1, len(motion_smooth))]
                    
                    if len(future_motion) < 3:
                        break
                    
                    # Motion should increase by a meaningful amount
                    motion_increase = future_motion[-1] - current_motion
                    
                    # Also check that most frames show increase
                    increases = sum(future_motion[j] < future_motion[j+1] if j+1 < len(future_motion) else False 
                                   for j in range(len(future_motion)-1))
                    
                    # Require: motion increases by at least 10% of range, and at least 50% of frames show increase
                    if motion_increase > motion_range * 0.10 and increases >= len(future_motion) * 0.5:
                        # Also verify motion was lower before this point
                        if i > 3:
                            prev_motion = motion_smooth[max(0, i-3):i]
                            if len(prev_motion) > 0 and np.mean(prev_motion) < current_motion * 1.2:
                                start_idx = i
                                break
        
        # Fallback: if local min approach didn't work, use threshold but require motion to be increasing
        if start_idx is None:
            # Compute simple derivative
            if len(motion_scores) > 1:
                motion_diff = np.diff(motion_scores)
            else:
                motion_diff = np.array([0])
            
            # Use a slightly higher threshold to avoid early detections
            adjusted_threshold = baseline + 0.20 * motion_range  # 20% instead of 15%
            
            for i in range(address_end_idx, peak_idx - 5):
                if i >= len(motion_diff):
                    continue
                
                # Motion should be above threshold
                if motion_scores[i] > adjusted_threshold:
                    # And should be increasing
                    if motion_diff[i] > 0:
                        # Check next few frames also show increase
                        if i + 3 < len(motion_diff):
                            if np.sum(motion_diff[i:i+3] > 0) >= 2:  # At least 2 of 3 increasing
                                window = motion_scores[i:i+self.min_sustained_frames]
                                if np.mean(window) > adjusted_threshold:
                                    start_idx = i
                                    break
        
        # Final fallback: original method
        if start_idx is None:
            for i in range(address_end_idx, len(motion_scores) - self.min_sustained_frames):
                if motion_scores[i] > start_threshold:
                    window = motion_scores[i:i+self.min_sustained_frames]
                    if np.mean(window) > start_threshold:
                        start_idx = i
                        break
        
        if start_idx is None:
            return None  # Couldn't detect start
        
        # Detect SWING END: motion drops and stays low after peak
        peak_idx = np.argmax(motion_scores)
        end_idx = None
        
        for i in range(peak_idx, len(motion_scores) - self.min_sustained_frames):
            if motion_scores[i] < end_threshold:
                # Check if motion stays low (golfer holding finish)
                window = motion_scores[i:i+self.min_sustained_frames]
                if np.mean(window) < end_threshold:
                    end_idx = i
                    break
        
        if end_idx is None:
            # Fallback: use last frame where motion is above threshold
            for i in range(len(motion_scores) - 1, peak_idx, -1):
                if motion_scores[i] < end_threshold:
                    end_idx = i
                    break
        
        if end_idx is None:
            return None  # Couldn't detect end
        
        # Convert to original frame indices
        start_frame = frame_indices[start_idx] if start_idx < len(frame_indices) else 0
        end_frame = frame_indices[end_idx] if end_idx < len(frame_indices) else frame_indices[-1]
        
        # Confidence based on how clear the signals are
        peak_value = motion_scores[peak_idx]
        confidence = min(1.0, (peak_value - baseline) / (motion_range + 1e-6))
        
        return (start_frame, end_frame, confidence)


class SwingBoundaryDetector:
    """
    Pose-based detector for golf swing boundaries using MediaPipe.
    
    Uses MediaPipe pose estimation to track golfer's body keypoints
    and detect swing start/end frames based on motion of wrists and shoulders.
    """
    
    def __init__(self, 
                 method='hybrid',  # Only hybrid method (optimized)
                 subsample_rate=2,  # Process every Nth frame (speed optimization)
                 optical_flow_weight=0.7,  # Weight for optical flow in fusion
                 frame_diff_weight=0.3,     # Weight for frame diff in fusion
                 start_percentile=95,        # Percentile for start detection (not used in new logic)
                 end_percentile=5,          # Percentile for end detection (not used in new logic)
                 min_sustained_frames=3,     # Frames above threshold for detection (reduced for sensitivity)
                 buffer_before=75,          # Frames before detected swing start
                 buffer_after=75,           # Frames after detected swing end
                 smoothing_sigma=1.5,       # Gaussian smoothing sigma (reduced to preserve edges)
                 smoothing_window=3,        # Moving average window (reduced to preserve edges)
                 use_adaptive_thresholds=True,  # Use percentile vs fixed
                 skip_early_percent=0,     # Skip first N% of video (0 = no skip, detect from start)
                 start_threshold_ratio=0.30,  # Start threshold as ratio from baseline to peak (0.3 = 30%, lower = catches earlier)
                 cache_dir=None):  # Directory to cache results
        """
        Initialize the swing boundary detector.
        
        Args:
            method: Detection method ('hybrid' only, optimized)
            subsample_rate: Process every Nth frame (2 = every 2nd frame, 2x speedup)
            optical_flow_weight: Weight for optical flow in fusion (0.0-1.0)
            frame_diff_weight: Weight for frame differencing in fusion (0.0-1.0)
            start_percentile: Percentile threshold for swing start detection
            end_percentile: Percentile threshold for swing end detection
            min_sustained_frames: Consecutive frames above threshold required
            buffer_before: Frames to add before detected swing start
            buffer_after: Frames to add after detected swing end
            smoothing_sigma: Gaussian smoothing sigma value
            smoothing_window: Moving average window size
            use_adaptive_thresholds: Use percentile-based adaptive thresholds
            cache_dir: Directory to cache detected boundaries (None = no caching)
        """
        self.method = method
        self.subsample_rate = subsample_rate
        self.optical_flow_weight = optical_flow_weight
        self.frame_diff_weight = frame_diff_weight
        self.start_percentile = start_percentile
        self.end_percentile = end_percentile
        self.min_sustained_frames = min_sustained_frames
        self.buffer_before = buffer_before
        self.buffer_after = buffer_after
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_window = smoothing_window
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.skip_early_percent = skip_early_percent
        self.start_threshold_ratio = start_threshold_ratio
        self.cache_dir = cache_dir
        
        # Normalize weights to sum to 1.0
        total_weight = optical_flow_weight + frame_diff_weight
        if total_weight > 0:
            self.optical_flow_weight = optical_flow_weight / total_weight
            self.frame_diff_weight = frame_diff_weight / total_weight
        else:
            self.optical_flow_weight = 0.7
            self.frame_diff_weight = 0.3
    
    def _compute_optical_flow(self, frame1, frame2):
        """
        Compute optical flow between two frames using Farneback method.
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            
        Returns:
            Motion magnitude per pixel (normalized by frame area)
        """
        # Optimized Farneback parameters
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2,
            None,
            pyr_scale=0.5,      # Pyramid scale (balance speed/accuracy)
            levels=3,            # Number of pyramid levels
            winsize=15,             # Averaging window size
            iterations=3,        # Iterations per level
            poly_n=5,            # Polynomial expansion neighborhood
            poly_sigma=1.2,      # Gaussian sigma for polynomial expansion
            flags=0
        )
        
        # Compute motion magnitude: sqrt(flow_x^2 + flow_y^2)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Sum magnitude across frame and normalize by area
        total_motion = np.sum(magnitude)
        frame_area = frame1.shape[0] * frame1.shape[1]
        normalized_motion = total_motion / frame_area if frame_area > 0 else 0.0
        
        return normalized_motion
    
    def _compute_frame_difference(self, frame1, frame2):
        """
        Compute frame difference between two frames.
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            
        Returns:
            Motion magnitude per pixel (normalized by frame area)
        """
        # Compute absolute difference
        diff = cv2.absdiff(frame1, frame2)
        
        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Sum pixel differences and normalize by area
        total_diff = np.sum(diff)
        frame_area = frame1.shape[0] * frame1.shape[1]
        normalized_diff = total_diff / frame_area if frame_area > 0 else 0.0
        
        return normalized_diff
    
    def _detect_slow_motion_regions(self, frame_diff_motion, window_size=15, threshold_percentile=25):
        """
        Detect slow motion regions in the video.
        
        Slow motion is characterized by:
        - Many frames with very small frame-to-frame differences
        - Consistent low motion over a window (but many frames)
        
        Uses RELATIVE motion (compared to max) rather than absolute thresholds,
        since slow motion swing might have higher motion than standing still,
        but still much lower than real-time motion would be.
        
        Args:
            frame_diff_motion: List of frame difference motion values
            window_size: Size of window to check for slow motion
            threshold_percentile: Percentile below which motion is considered slow
            
        Returns:
            List of (start_idx, end_idx) tuples for slow motion regions (in frame_indices space)
        """
        if len(frame_diff_motion) < window_size:
            return []
        
        frame_diff_array = np.array(frame_diff_motion)
        
        # Use RELATIVE threshold: compare to maximum motion in video
        # This catches slow motion swing (which has motion, but less than real-time would)
        max_motion = np.max(frame_diff_array)
        median_motion = np.median(frame_diff_array)
        
        # Slow motion threshold: if motion is significantly below max, it's likely slow motion
        # Use a combination of percentile and relative to max
        percentile_threshold = np.percentile(frame_diff_array, threshold_percentile)
        relative_threshold = max_motion * 0.3  # If motion is less than 30% of max, likely slow motion
        
        # Use the higher of the two thresholds to be more inclusive
        # (catches both standing still AND slow motion swing)
        slow_threshold = max(percentile_threshold, relative_threshold)
        
        print(f"   DEBUG: Slow motion detection - max: {max_motion:.4f}, median: {median_motion:.4f}, threshold: {slow_threshold:.4f}")
        
        slow_motion_regions = []
        in_slow_motion = False
        slow_start = None
        
        for i in range(len(frame_diff_motion) - window_size + 1):
            window = frame_diff_motion[i:i+window_size]
            window_mean = np.mean(window)
            
            # Check if window is consistently low motion (slow motion)
            # Use more lenient threshold - if most frames are below threshold, it's slow motion
            below_threshold_count = np.sum(window < slow_threshold * 1.5)
            if below_threshold_count >= window_size * 0.7:  # At least 70% of window is low motion
                if not in_slow_motion:
                    in_slow_motion = True
                    slow_start = i
            else:
                if in_slow_motion:
                    in_slow_motion = False
                    # Only add if region is substantial (at least window_size frames)
                    if i - slow_start >= window_size:
                        slow_motion_regions.append((slow_start, i + window_size - 1))
        
        # Close any open slow motion region
        if in_slow_motion and len(frame_diff_motion) - slow_start >= window_size:
            slow_motion_regions.append((slow_start, len(frame_diff_motion) - 1))
        
        # If we found multiple slow motion regions, treat everything from first to last as one continuous region
        # (since slow motion doesn't have breaks - it's continuous)
        if len(slow_motion_regions) > 1:
            first_slow_start = min(reg[0] for reg in slow_motion_regions)
            last_slow_end = max(reg[1] for reg in slow_motion_regions)
            slow_motion_regions = [(first_slow_start, last_slow_end)]
            print(f"   DEBUG: Merged {len(slow_motion_regions)} slow motion regions into one continuous region: [{first_slow_start}-{last_slow_end}]")
        
        return slow_motion_regions
    
    def _detect_swing_phases(self, motion_series, frame_indices, fps):
        """
        Detect key swing phases from motion profile.
        
        Args:
            motion_series: Smoothed motion values
            frame_indices: Original frame indices corresponding to motion values
            fps: Video frame rate
            
        Returns:
            dict with 'address', 'top', 'impact', 'finish' frame indices
        """
        if len(motion_series) == 0 or len(frame_indices) == 0:
            return None
        
        # Find peak (impact)
        peak_idx = np.argmax(motion_series)
        impact_frame = frame_indices[peak_idx] if peak_idx < len(frame_indices) else frame_indices[-1]
        
        # Find start of motion (address) - first significant rise
        if len(motion_series) > 1:
            motion_derivative = np.diff(motion_series)
            derivative_threshold = np.percentile(motion_derivative, 70)
            
            address_idx = 0
            for i in range(len(motion_derivative)):
                if motion_derivative[i] > derivative_threshold:
                    address_idx = i
                    break
            address_frame = frame_indices[address_idx] if address_idx < len(frame_indices) else frame_indices[0]
        else:
            address_frame = frame_indices[0] if len(frame_indices) > 0 else 0
        
        # Find top of backswing (local max before impact)
        if peak_idx > 0:
            before_impact = motion_series[:peak_idx]
            if len(before_impact) > 0:
                top_idx = np.argmax(before_impact)
                top_frame = frame_indices[top_idx] if top_idx < len(frame_indices) else address_frame
            else:
                top_frame = address_frame
        else:
            top_frame = address_frame
        
        # Find finish (motion drops after impact)
        if peak_idx < len(motion_series) - 1:
            after_impact = motion_series[peak_idx:]
            baseline = np.median(motion_series)
            finish_idx = peak_idx
            # Look for where motion drops significantly after impact
            for i in range(len(after_impact)):
                if after_impact[i] < baseline * 1.2:
                    finish_idx = peak_idx + i
                    # Make sure we have at least some frames after impact
                    if finish_idx > peak_idx + 5:  # At least 5 frames after impact
                        break
            # Don't let finish be too close to the end (unless it really is at the end)
            if finish_idx < len(frame_indices) - 1:
                finish_frame = frame_indices[finish_idx] if finish_idx < len(frame_indices) else frame_indices[-1]
            else:
                finish_frame = frame_indices[-1] if len(frame_indices) > 0 else impact_frame
        else:
            finish_frame = frame_indices[-1] if len(frame_indices) > 0 else impact_frame
        
        # Validate phases make sense
        if address_frame >= finish_frame:
            # Invalid phases - use peak-based estimation
            print(f"   DEBUG: Invalid phases detected (address >= finish), using peak-based estimation")
            if len(frame_indices) > 0:
                peak_frame = frame_indices[peak_idx] if peak_idx < len(frame_indices) else frame_indices[-1]
                address_frame = max(0, peak_frame - total_frames // 3) if 'total_frames' in locals() else max(0, peak_frame - 100)
                finish_frame = min(frame_indices[-1], peak_frame + total_frames // 3) if 'total_frames' in locals() else min(frame_indices[-1], peak_frame + 100)
        
        phases = {
            'address': address_frame,
            'top': top_frame,
            'impact': impact_frame,
            'finish': finish_frame
        }
        
        # Debug output
        if len(frame_indices) > 0:
            total_frames_for_phases = frame_indices[-1] + 1
            phase_span = finish_frame - address_frame
            span_ratio = phase_span / total_frames_for_phases if total_frames_for_phases > 0 else 1.0
            print(f"   DEBUG: Detected phases - Address: {address_frame}, Top: {top_frame}, Impact: {impact_frame}, Finish: {finish_frame}")
            print(f"      Phase span: {phase_span} frames ({span_ratio*100:.1f}% of video)")
        
        return phases
    
    def _check_biomechanical_timing(self, phases, fps):
        """
        Check if phase durations suggest slow motion.
        
        Biomechanical limits (pro/amateur ranges):
        - Backswing: 0.6-1.2s
        - Downswing: 0.18-0.35s
        - Total swing: 1.0-1.6s
        
        Args:
            phases: dict with 'address', 'top', 'impact', 'finish' frame indices
            fps: Video frame rate
            
        Returns:
            (is_slow_motion, confidence, details_dict)
        """
        if phases is None:
            return False, 0.0, {}
        
        backswing_duration = (phases['top'] - phases['address']) / fps if fps > 0 else 0
        downswing_duration = (phases['impact'] - phases['top']) / fps if fps > 0 else 0
        total_duration = (phases['finish'] - phases['address']) / fps if fps > 0 else 0
        
        # Biomechanical limits (generous upper bounds)
        MAX_DOWNSWING = 0.4  # seconds (normal is 0.18-0.35s)
        MAX_BACKSWING = 1.5  # seconds (normal is 0.6-1.2s)
        MAX_TOTAL = 2.0      # seconds (normal is 1.0-1.6s)
        
        slow_motion_flags = []
        
        if downswing_duration > MAX_DOWNSWING:
            slow_motion_flags.append(f"Downswing too long: {downswing_duration:.2f}s (max: {MAX_DOWNSWING}s)")
        
        if backswing_duration > MAX_BACKSWING:
            slow_motion_flags.append(f"Backswing too long: {backswing_duration:.2f}s (max: {MAX_BACKSWING}s)")
        
        if total_duration > MAX_TOTAL:
            slow_motion_flags.append(f"Total swing too long: {total_duration:.2f}s (max: {MAX_TOTAL}s)")
        
        # Need 2+ flags to be confident it's slow motion
        is_slow_motion = len(slow_motion_flags) >= 2
        
        # Confidence based on how many flags and how far over limits
        confidence = min(1.0, len(slow_motion_flags) / 3.0)
        if is_slow_motion:
            # Increase confidence if durations are way over limits
            overage = max(
                max(0, downswing_duration - MAX_DOWNSWING) / MAX_DOWNSWING,
                max(0, backswing_duration - MAX_BACKSWING) / MAX_BACKSWING,
                max(0, total_duration - MAX_TOTAL) / MAX_TOTAL
            )
            confidence = min(1.0, confidence + overage * 0.3)
        
        return is_slow_motion, confidence, {
            'backswing_duration': backswing_duration,
            'downswing_duration': downswing_duration,
            'total_duration': total_duration,
            'flags': slow_motion_flags
        }
    
    def _check_acceleration_profile(self, motion_series):
        """
        Real swings have sharp acceleration spikes near impact.
        Slow motion has smooth, gradual increases.
        
        Args:
            motion_series: Smoothed motion values
            
        Returns:
            (is_slow_motion, accel_ratio)
        """
        if len(motion_series) < 3:
            return False, 0.0
        
        motion_derivative = np.diff(motion_series)
        acceleration = np.diff(motion_derivative)
        
        # Find max acceleration
        max_accel = np.max(np.abs(acceleration))
        median_accel = np.median(np.abs(acceleration))
        
        # Real swings: max acceleration >> median (sharp spike)
        # Slow motion: max acceleration ≈ median (smooth)
        accel_ratio = max_accel / (median_accel + 1e-6)
        
        # If ratio is low, likely slow motion (smooth acceleration)
        is_slow_motion = accel_ratio < 3.0
        
        return is_slow_motion, accel_ratio
    
    def _detect_slow_motion_regions_hybrid(self, frame_diff_motion, motion_series, frame_indices, fps):
        """
        Hybrid slow motion detection using:
        1. Biomechanical timing (primary) - most reliable
        2. Acceleration profile (secondary)
        3. Motion magnitude analysis (fallback)
        
        Args:
            frame_diff_motion: Frame difference motion values
            motion_series: Smoothed combined motion values
            frame_indices: Original frame indices
            fps: Video frame rate
            
        Returns:
            List of (start_idx, end_idx) tuples for slow motion regions
        """
        # First, try biomechanical timing (most reliable)
        phases = self._detect_swing_phases(motion_series, frame_indices, fps)
        
        if phases is not None:
            is_slow_motion_timing, timing_confidence, timing_details = self._check_biomechanical_timing(phases, fps)
            
            # Check acceleration profile as secondary signal
            is_slow_motion_accel, accel_ratio = self._check_acceleration_profile(motion_series)
            
            # Combine signals: timing is primary, acceleration is supporting
            if is_slow_motion_timing or (is_slow_motion_accel and timing_confidence > 0.3):
                print(f"   DEBUG: Slow motion detected via biomechanical timing:")
                print(f"      Backswing: {timing_details['backswing_duration']:.2f}s")
                print(f"      Downswing: {timing_details['downswing_duration']:.2f}s")
                print(f"      Total: {timing_details['total_duration']:.2f}s")
                print(f"      Acceleration ratio: {accel_ratio:.2f} (low = smooth = slow motion)")
                for flag in timing_details['flags']:
                    print(f"      ⚠️  {flag}")
                
                # Mark entire swing region as slow motion
                # Find indices in frame_indices that correspond to phase frames
                slow_start_idx = 0
                slow_end_idx = len(frame_indices) - 1
                
                for i, f in enumerate(frame_indices):
                    if f >= phases['address']:
                        slow_start_idx = i
                        break
                
                for i, f in enumerate(frame_indices):
                    if f >= phases['finish']:
                        slow_end_idx = i
                        break
                
                return [(slow_start_idx, slow_end_idx)]
            else:
                print(f"   DEBUG: Biomechanical timing normal (backswing: {timing_details['backswing_duration']:.2f}s, downswing: {timing_details['downswing_duration']:.2f}s)")
        
        # Fallback to motion magnitude analysis
        print(f"   DEBUG: Using motion magnitude analysis (biomechanical timing not available or normal)")
        return self._detect_slow_motion_regions(frame_diff_motion)
    
    def _smooth_motion_series(self, motion_series):
        """
        Apply advanced smoothing to motion series.
        
        Args:
            motion_series: 1D array of motion values
            
        Returns:
            Smoothed motion series
        """
        if len(motion_series) < 3:
            return motion_series
        
        # Apply Gaussian smoothing
        smoothed = ndimage.gaussian_filter1d(motion_series, sigma=self.smoothing_sigma)
        
        # Apply moving average
        if self.smoothing_window > 1 and len(smoothed) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            smoothed = np.convolve(smoothed, kernel, mode='same')
        
        return smoothed
    
    def _detect_boundaries_adaptive(self, motion_series, frame_indices, slow_motion_regions=None, total_original_frames=None):
        """
        Detect swing boundaries using adaptive percentile-based thresholds.
        
        Args:
            motion_series: Smoothed motion values
            frame_indices: Original frame indices corresponding to motion values
            slow_motion_regions: List of (start_idx, end_idx) tuples for slow motion regions
            total_original_frames: Total frames in original video (for validation)
            
        Returns:
            (start_frame, end_frame, confidence_score)
        """
        # Get total frames for validation
        if total_original_frames is None:
            total_original_frames = frame_indices[-1] + 1 if len(frame_indices) > 0 else 1000
        if len(motion_series) < self.min_sustained_frames:
            # Not enough data - use peak position to estimate start
            if len(motion_series) > 0 and len(frame_indices) > 0:
                peak_idx = np.argmax(motion_series)
                # Start at 30% of way to peak
                start_idx = max(1, peak_idx // 3)  # Never 0
                if start_idx < len(frame_indices):
                    start_frame = frame_indices[start_idx]
                else:
                    # Use peak-based calculation
                    peak_frame = frame_indices[peak_idx] if peak_idx < len(frame_indices) else frame_indices[-1]
                    peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                    start_frame = int(peak_frame * peak_based_start_ratio)
                    if start_frame == 0:
                        start_frame = max(1, peak_frame // 5)
                # Don't use last frame - use peak-based end
                peak_frame = frame_indices[peak_idx] if peak_idx < len(frame_indices) else frame_indices[-1]
                # End should be after peak, but not at the very end
                end_frame = min(peak_frame + 100, total_original_frames - 1, frame_indices[-1] if len(frame_indices) > 0 else total_original_frames - 1)
                return (start_frame, end_frame, 0.0)
            # Last resort - use middle portion
            if len(frame_indices) > 0:
                mid_point = len(frame_indices) // 2
                start_frame = frame_indices[mid_point // 2] if mid_point // 2 < len(frame_indices) else frame_indices[0]
                end_frame = frame_indices[-1]
                return (start_frame, end_frame, 0.0)
            return (1, 1, 0.0)  # Absolute last resort
        
        # Find peak motion (swing apex)
        # If slow motion regions exist and motion has been boosted, the peak should be in slow motion
        # (since that's where the actual swing is, just slowed down)
        # After boosting, the peak should naturally be in the slow motion section
        peak_idx = np.argmax(motion_series)
        
        # Verify peak location
        if slow_motion_regions:
            peak_in_slow_motion = any(slow_start <= peak_idx <= slow_end 
                                     for slow_start, slow_end in slow_motion_regions)
            if peak_in_slow_motion:
                print(f"   DEBUG: Peak found in slow motion region at index {peak_idx}")
            else:
                print(f"   DEBUG: Peak found in real-time section at index {peak_idx}")
        peak_motion = motion_series[peak_idx]
        
        # Compute statistics for adaptive thresholds
        mean_motion = np.mean(motion_series)
        std_motion = np.std(motion_series)
        median_motion = np.median(motion_series)
        
        # Use a lower threshold for start - we want to catch when motion begins to rise
        # Use a percentage of the way from baseline to peak
        baseline_motion = median_motion  # Use median as baseline (more robust than mean)
        motion_range = peak_motion - baseline_motion
        
        if motion_range < 1e-6:
            # No significant motion - use peak position to estimate start
            # Even with low motion range, peak_idx should still be valid
            # Start at 30% of way to peak
            start_idx = max(1, peak_idx // 3)  # Never 0
            if start_idx < len(frame_indices):
                start_frame = frame_indices[start_idx]
            else:
                # Use peak-based calculation
                peak_frame = frame_indices[peak_idx] if peak_idx < len(frame_indices) else frame_indices[-1]
                peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                start_frame = int(peak_frame * peak_based_start_ratio)
                if start_frame == 0:
                    start_frame = max(1, peak_frame // 5)
            end_frame = frame_indices[-1] if len(frame_indices) > 0 else total_original_frames - 1
            return (start_frame, end_frame, 0.0)
        
        # Start threshold: when motion reaches specified ratio from baseline to peak
        # Higher ratio = more conservative (catches later in swing)
        start_threshold = baseline_motion + self.start_threshold_ratio * motion_range
        
        # End threshold: when motion drops back to 40% of the way from baseline to peak
        # This catches when the swing motion ends
        end_threshold = baseline_motion + 0.40 * motion_range
        
        # IMPORTANT: Look for TRANSITION from low motion to high motion
        # Don't just find when motion crosses threshold - find when it TRANSITIONS
        # Calculate a "low motion" threshold (e.g., 10% of way to peak)
        low_motion_threshold = baseline_motion + 0.10 * motion_range
        
        # Detect start: Find TRANSITION from low motion to high motion
        # This is more reliable than just crossing a threshold
        start_idx = None
        
        # Calculate motion derivative (rate of change) and acceleration (change in rate)
        if len(motion_series) > 2:
            motion_derivative = np.diff(motion_series)
            motion_acceleration = np.diff(motion_derivative)  # Second derivative
            
            # Find where motion transitions from low to high
            # Look for: low motion before, then rapid increase to high motion
            search_start = max(0, int(len(motion_series) * self.skip_early_percent / 100))
            
            # Look for transition: motion was LOW for a period, then RAPIDLY increases to HIGH
            # Need at least 5 frames of history to check
            lookback_window = 8  # Check last 8 frames for low motion
            min_search_position = search_start + lookback_window  # Only respect skip_early_percent if set
            for i in range(min_search_position, peak_idx - 2):
                if i < len(motion_acceleration) and i < len(motion_derivative):
                    # Check if we're transitioning from low to high motion
                    # REQUIRE: Previous frames (last 5-8) should be consistently LOW
                    if i >= lookback_window:
                        prev_motions = motion_series[i-lookback_window:i]
                        # At least 6 out of 8 previous frames should be low
                        low_count = np.sum(prev_motions < low_motion_threshold)
                        prev_motion_low = low_count >= 6
                        
                        # Current frame should be HIGH (above start threshold)
                        current_motion_high = motion_series[i] >= start_threshold
                        
                        # Motion should be INCREASING significantly
                        if i < len(motion_derivative):
                            recent_derivative = motion_derivative[max(0, i-3):i+1] if i >= 3 else motion_derivative[:i+1]
                            is_increasing = np.mean(recent_derivative) > np.percentile(motion_derivative, 60) if len(motion_derivative) > 0 else False
                        else:
                            is_increasing = False
                        
                        # Motion should be ACCELERATING (rapidly increasing)
                        if i < len(motion_acceleration):
                            is_accelerating = motion_acceleration[i] > np.percentile(motion_acceleration, 70) if len(motion_acceleration) > 0 else False
                        else:
                            is_accelerating = False
                        
                        # ALL conditions must be met for a valid transition
                        if prev_motion_low and current_motion_high and (is_accelerating or is_increasing):
                            # Found transition! Verify it's sustained (stays high)
                            if i + self.min_sustained_frames <= len(motion_series):
                                sustained = True
                                for j in range(i, min(i + self.min_sustained_frames, len(motion_series))):
                                    if motion_series[j] < start_threshold * 0.80:
                                        sustained = False
                                        break
                                if sustained:
                                    # This is the transition point - the swing start
                                    start_idx = i
                                    break
        
        # Fallback: if no transition found, look for when motion crosses from below to above threshold
        # But require a period of low motion before
        if start_idx is None:
            if len(motion_series) > 1:
                motion_derivative = np.diff(motion_series)
                derivative_threshold = np.percentile(motion_derivative, 70)  # 70th percentile
                
                search_start = max(0, int(len(motion_series) * self.skip_early_percent / 100))
                # Look for crossing: was below threshold for a while, now above and increasing
                for i in range(search_start + 5, peak_idx):
                    if i < len(motion_derivative) and i >= 5:
                        # Check that previous 5 frames were mostly below threshold
                        prev_frames = motion_series[i-5:i]
                        mostly_below = np.sum(prev_frames < start_threshold) >= 4  # At least 4 of 5 below
                        
                        # Current frame is above threshold
                        now_above = motion_series[i] >= start_threshold
                        
                        # Motion is increasing
                        is_increasing = motion_derivative[i-1] > derivative_threshold if i > 0 else False
                        
                        if mostly_below and now_above and is_increasing:
                            # Check if it stays above threshold
                            if i + self.min_sustained_frames <= len(motion_series):
                                sustained = True
                                for j in range(i, min(i + self.min_sustained_frames, len(motion_series))):
                                    if motion_series[j] < start_threshold * 0.85:
                                        sustained = False
                                        break
                                if sustained:
                                    start_idx = i
                                    break
        
        # Last resort: find first significant increase in motion (not just crossing threshold)
        if start_idx is None:
            search_start = max(0, int(len(motion_series) * self.skip_early_percent / 100))
            if len(motion_series) > 1:
                motion_derivative = np.diff(motion_series)
                # Look for when derivative becomes significantly positive (motion starts increasing)
                derivative_threshold = np.percentile(motion_derivative, 60)
                for i in range(search_start, peak_idx):
                    if (i < len(motion_derivative) and 
                        motion_derivative[i] > derivative_threshold and
                        motion_series[i] > baseline_motion + 0.20 * motion_range):
                        # Motion is increasing significantly and is above baseline
                        start_idx = i
                        break
        
        # Detect end: Find when motion drops back down after peak (searching forward from peak)
        end_idx = None
        for i in range(peak_idx, len(motion_series)):
            if motion_series[i] <= end_threshold:
                # Check if it stays below threshold for sustained period
                if i + self.min_sustained_frames <= len(motion_series):
                    sustained = True
                    for j in range(i, min(i + self.min_sustained_frames, len(motion_series))):
                        if motion_series[j] > end_threshold * 1.1:  # Allow small tolerance
                            sustained = False
                            break
                    if sustained:
                        end_idx = min(len(motion_series) - 1, i + self.min_sustained_frames - 1)
                        break
        
        # If no end found, look for last frame above baseline
        if end_idx is None:
            for i in range(len(motion_series) - 1, peak_idx, -1):
                if motion_series[i] > baseline_motion + 0.1 * motion_range:
                    end_idx = min(len(motion_series) - 1, i + 5)  # Go forward 5 frames
                    break
        
        # Fallback: use peak-centered window if detection failed
        if start_idx is None:
            # Use a position that's a percentage of the way to peak (before the peak)
            # Higher start_threshold_ratio means we want to start later, so use a higher percentage
            # If threshold is 0.7, we want to start at ~30% before peak (70% of way to peak)
            # If threshold is 0.3, we want to start at ~70% before peak (30% of way to peak)
            # Formula: start_position = (1 - start_threshold_ratio) * peak_idx
            peak_based_start_ratio = 1.0 - self.start_threshold_ratio
            calculated_start = int(peak_idx * peak_based_start_ratio)
            
            # Ensure it's not 0 (unless peak is at 0, which shouldn't happen)
            if calculated_start == 0 and peak_idx > 0:
                # Use at least 20% of way to peak
                calculated_start = max(1, peak_idx // 5)
            
            # If skip_early_percent is set, ensure we're at least that far in
            if self.skip_early_percent > 0:
                min_start = max(1, int(len(motion_series) * self.skip_early_percent / 100))
                calculated_start = max(min_start, calculated_start)
            
            print(f"   DEBUG FALLBACK: Using peak-based start at index {calculated_start} (peak at {peak_idx}, video length {len(motion_series)})")
            start_idx = calculated_start
        
        if end_idx is None:
            # End should be well after peak - use 30% of video after peak
            end_idx = min(len(motion_series) - 1, peak_idx + len(motion_series) // 3)
        
        # Ensure start < end
        if start_idx >= end_idx:
            # If detection failed, use a reasonable window around peak
            window_size = max(30, len(motion_series) // 4)
            # Start should be before peak, use peak-based calculation
            peak_based_start_ratio = 1.0 - self.start_threshold_ratio
            calculated_start = int(peak_idx * peak_based_start_ratio)
            if calculated_start == 0 and peak_idx > 0:
                calculated_start = max(1, peak_idx // 5)
            start_idx = max(calculated_start, peak_idx - window_size // 2)
            end_idx = min(len(motion_series) - 1, peak_idx + window_size // 2)
        
        # Only enforce skip_early_percent if it's explicitly set (> 0)
        if self.skip_early_percent > 0:
            min_start_frame = max(1, int(len(motion_series) * self.skip_early_percent / 100))
            if start_idx < min_start_frame:
                start_idx = min_start_frame
        
        # Convert back to original frame indices
        if start_idx < len(frame_indices):
            start_frame = frame_indices[start_idx]
        else:
            # If out of bounds, use peak-based position in original frame space
            if len(frame_indices) > 0 and peak_idx < len(frame_indices):
                peak_frame = frame_indices[peak_idx]
                # Calculate start as percentage before peak
                peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                start_frame = int(peak_frame * peak_based_start_ratio)
                if start_frame == 0 and peak_frame > 0:
                    start_frame = max(1, peak_frame // 5)
            elif len(frame_indices) > 0:
                # Use peak frame if available, otherwise use a position based on last frame
                if peak_idx < len(frame_indices):
                    peak_frame = frame_indices[peak_idx]
                    peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                    start_frame = int(peak_frame * peak_based_start_ratio)
                    if start_frame == 0:
                        start_frame = max(1, peak_frame // 5)
                else:
                    # Use percentage of last frame
                    last_frame = frame_indices[-1]
                    peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                    start_frame = int(last_frame * peak_based_start_ratio)
                    if start_frame == 0:
                        start_frame = max(1, last_frame // 5)
            else:
                # No frame indices - this shouldn't happen, but use 1 as minimum
                start_frame = 1
        
        if end_idx < len(frame_indices):
            end_frame = frame_indices[end_idx]
        else:
            # Don't default to last frame - use peak-based estimate
            if 'peak_idx' in locals() and peak_idx < len(frame_indices):
                peak_frame = frame_indices[peak_idx]
                # End should be after peak, but reasonable (not entire video)
                end_frame = min(peak_frame + 150, total_original_frames - 1)
            else:
                end_frame = min(total_original_frames - 1, frame_indices[-1] if len(frame_indices) > 0 else total_original_frames - 1)
        
        # Final validation: only enforce skip_early_percent if explicitly set
        if self.skip_early_percent > 0 and len(frame_indices) > 0:
            # Calculate minimum frame in original video space
            total_original_frames = frame_indices[-1] + 1 if len(frame_indices) > 0 else 1
            min_start_frame_original = max(1, int(total_original_frames * self.skip_early_percent / 100))
            if start_frame < min_start_frame_original:
                # Find the closest frame_indices entry that's >= min_start_frame_original
                for idx, orig_frame in enumerate(frame_indices):
                    if orig_frame >= min_start_frame_original:
                        start_frame = orig_frame
                        break
                # If still not found, use the first frame after skip_early_percent
                if start_frame < min_start_frame_original:
                    start_frame = min_start_frame_original
        
        # Compute confidence score (based on how clear the peak is and motion range)
        peak_ratio = (peak_motion - baseline_motion) / (std_motion + 1e-6)
        confidence = min(1.0, peak_ratio / 3.0)  # Normalize
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        # Final check: only enforce skip if explicitly set
        if self.skip_early_percent > 0 and start_frame == 0 and len(frame_indices) > 0:
            # Calculate what the minimum should be based on the last frame index
            # frame_indices are subsampled, so estimate total frames
            estimated_total = frame_indices[-1] * self.subsample_rate if len(frame_indices) > 0 else 100
            min_frame = max(1, int(estimated_total * self.skip_early_percent / 100))
            # Find closest frame_indices entry >= min_frame
            for orig_frame in frame_indices:
                if orig_frame >= min_frame:
                    start_frame = orig_frame
                    break
            # If still 0, force it to at least the first non-zero frame_indices entry
            if start_frame == 0 and len(frame_indices) > 1:
                start_frame = frame_indices[1] if frame_indices[1] > 0 else min_frame
        
        # ABSOLUTE FINAL SAFETY CHECK: start_frame must NEVER be 0 (unless video is 1 frame)
        if start_frame == 0 and total_original_frames > 1:
            # Use peak-based calculation as last resort
            if len(frame_indices) > 0 and len(motion_series) > 0:
                peak_idx = np.argmax(motion_series)
                if peak_idx < len(frame_indices):
                    peak_frame = frame_indices[peak_idx]
                    peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                    start_frame = int(peak_frame * peak_based_start_ratio)
                    if start_frame == 0:
                        start_frame = max(1, peak_frame // 5)
                else:
                    # Use percentage of last frame
                    last_frame = frame_indices[-1]
                    peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                    start_frame = int(last_frame * peak_based_start_ratio)
                    if start_frame == 0:
                        start_frame = max(1, last_frame // 5)
            else:
                # No frame indices - use percentage of total frames
                peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                start_frame = int(total_original_frames * peak_based_start_ratio)
                if start_frame == 0:
                    start_frame = max(1, total_original_frames // 5)
        
        # Final validation: ensure boundaries don't span entire video
        if start_frame == 0 or end_frame >= total_original_frames * 0.95:
            # Invalid boundaries - use peak-based fallback
            if len(frame_indices) > 0 and len(motion_series) > 0:
                peak_idx = np.argmax(motion_series)
                if peak_idx < len(frame_indices):
                    peak_frame = frame_indices[peak_idx]
                    peak_based_start_ratio = 1.0 - self.start_threshold_ratio
                    start_frame = int(peak_frame * peak_based_start_ratio)
                    if start_frame == 0:
                        start_frame = max(1, peak_frame // 5)
                    # End should be after peak, but reasonable
                    end_frame = min(peak_frame + 150, total_original_frames - 1)
                    confidence = 0.3  # Lower confidence for fallback
                    print(f"   DEBUG: Boundary validation failed, using peak-based fallback: {start_frame}-{end_frame}")
        
        return (start_frame, end_frame, confidence)
    
    def _load_cached_boundaries(self, video_path):
        """Load cached boundaries if available."""
        if self.cache_dir is None:
            return None
        
        video_id = osp.splitext(osp.basename(video_path))[0]
        cache_path = osp.join(self.cache_dir, f'{video_id}_boundaries.pkl')
        
        if osp.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_cached_boundaries(self, video_path, start_frame, end_frame, confidence):
        """Save detected boundaries to cache."""
        if self.cache_dir is None:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        video_id = osp.splitext(osp.basename(video_path))[0]
        cache_path = osp.join(self.cache_dir, f'{video_id}_boundaries.pkl')
        
        try:
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump((start_frame, end_frame, confidence), f)
        except Exception:
            pass  # Silently fail if cache write fails
    
    def detect_swing_boundaries(self, video_path):
        """
        Detect swing boundaries using pose-based detection.
        
        Args:
            video_path: Path to video file
            
        Returns:
            (start_frame, end_frame, confidence_score)
            start_frame and end_frame include buffer padding
            
        Raises:
            ValueError: If MediaPipe is not available or pose detection fails
        """
        # Check cache first
        cached = self._load_cached_boundaries(video_path)
        if cached is not None:
            return cached
        
        # Pose-based detection (required)
        if not MEDIAPIPE_AVAILABLE:
            raise ValueError("MediaPipe is required for pose-based detection. Please install: pip install mediapipe")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                pose_detector = PoseBasedSwingDetector(
                    start_motion_threshold=0.15,
                    end_motion_threshold=0.10,
                    min_sustained_frames=8
                )
                
                pose_result = pose_detector.detect_swing_boundaries(
                    video_path, fps, subsample_rate=self.subsample_rate
                )
                
                if pose_result is not None:
                    start_frame, end_frame, confidence = pose_result
                    # Add buffers
                    start_frame = max(0, start_frame - self.buffer_before)
                    end_frame = min(total_frames - 1, end_frame + self.buffer_after)
                    
                    # Validate boundaries are reasonable
                    # Must have valid start < end and not span entire video
                    if (start_frame < end_frame and 
                        start_frame >= 0 and 
                        end_frame < total_frames and
                        (end_frame - start_frame) < total_frames * 0.95 and
                        (end_frame - start_frame) > 10):  # At least 10 frames
                        print(f"   ✅ Pose-based detection successful: frames {start_frame}-{end_frame} ({end_frame-start_frame+1} frames)")
                        result = (start_frame, end_frame, confidence)
                        self._save_cached_boundaries(video_path, *result)
                        return result
                    else:
                        raise ValueError(f"Pose-based detection returned invalid boundaries (start: {start_frame}, end: {end_frame}, span: {end_frame-start_frame}/{total_frames})")
                else:
                    raise ValueError("Pose-based detection failed - no swing boundaries detected")
            else:
                raise ValueError(f"Could not open video for pose detection: {video_path}")
        except ValueError:
            raise  # Re-raise ValueError as-is
        except Exception as e:
            raise ValueError(f"Pose-based detection error: {e}")
