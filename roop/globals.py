# roop/globals.py

from typing import List, Optional

# Path-related attributes
source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None
headless: Optional[bool] = None

# Frame processor attributes
frame_processors: List[str] = []

# Processing options
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = None
reference_face_position: Optional[int] = None
reference_frame_number: Optional[int] = None
similar_face_distance: Optional[float] = None
temp_frame_format: Optional[str] = None
temp_frame_quality: Optional[int] = None
output_video_encoder: Optional[str] = None
output_video_quality: Optional[int] = None
max_memory: Optional[int] = None
execution_providers: List[str] = []
execution_threads: Optional[int] = None
log_level: str = 'error'

# New attributes for glitch mitigation and enhancements
occlusion_threshold: Optional[float] = None  # Default 0.6 from CLI
side_pose_adjust: Optional[bool] = None
face_tracking: Optional[bool] = None
max_faces: Optional[int] = None  # Default 1 from CLI
refine_landmarks: Optional[bool] = None
deform_threshold: Optional[float] = None  # Default 0.3 from CLI
mouth_mask: Optional[bool] = None
erode_mask: Optional[float] = None  # Default 0.2 from CLI
enhance_faces: Optional[bool] = None
