# roop/capturer.py

from typing import Optional
import cv2
import os

from roop.typing import Frame

def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    """
    Retrieve a specific frame from a video by frame number.
    
    Args:
        video_path (str): Path to the video file.
        frame_number (int): Frame number to retrieve (0-based indexing).
    
    Returns:
        Optional[Frame]: The frame as a numpy array, or None if failed.
    """
    if not os.path.exists(video_path):
        return None
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None
    
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= frame_total or frame_number < 0:
        frame_number = max(0, frame_total - 1)  # Default to last frame if out of range
    
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    capture.release()
    
    return frame if has_frame else None

def get_video_frame_total(video_path: str) -> int:
    """
    Get the total number of frames in a video.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        int: Total number of frames, or 0 if failed.
    """
    if not os.path.exists(video_path):
        return 0
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return 0
    
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    
    return video_frame_total
