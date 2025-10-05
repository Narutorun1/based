# roop/face_reference.py

from typing import Optional
import threading

from roop.typing import Face

# Global face reference with thread lock
FACE_REFERENCE = None
FACE_REFERENCE_LOCK = threading.Lock()

def get_face_reference() -> Optional[Face]:
    """
    Retrieve the current face reference.
    
    Returns:
        Optional[Face]: The stored face reference, or None if not set.
    """
    with FACE_REFERENCE_LOCK:
        return FACE_REFERENCE

def set_face_reference(face: Face) -> None:
    """
    Set the global face reference to the provided face.
    
    Args:
        face (Face): The face object to set as the reference.
    
    Raises:
        ValueError: If face is None.
    """
    if face is None:
        raise ValueError("Face reference cannot be set to None")
    with FACE_REFERENCE_LOCK:
        global FACE_REFERENCE
        FACE_REFERENCE = face

def clear_face_reference() -> None:
    """
    Clear the global face reference, setting it to None.
    """
    with FACE_REFERENCE_LOCK:
        global FACE_REFERENCE
        FACE_REFERENCE = None
