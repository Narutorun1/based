# roop/__init__.py
from .run import run
from .ui import init
from .globals import *
from .predictor import get_landmarks, clear_predictor
from .face_analyser import get_one_face
from .capturer import get_video_frame, get_video_frame_total
from .face_reference import get_face_reference, set_face_reference, clear_face_reference
from .processors.frame.core import get_frame_processors_modules

__all__ = [
    'run',
    'init',
    'get_landmarks',
    'clear_predictor',
    'get_one_face',
    'get_video_frame',
    'get_video_frame_total',
    'get_face_reference',
    'set_face_reference',
    'clear_face_reference',
    'get_frame_processors_modules'
]
# if not hasattr(globals(), 'source_path'):
#     globals.source_path = None  # Add defaults if needed

