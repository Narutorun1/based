import threading
from typing import Any, Optional, List
import insightface
import numpy as np
import cv2
import mediapipe as mp
from roop.globals import execution_providers, similar_face_distance
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()
MP_FACE_MESH = None  # MediaPipe for enhanced detection

def initialize_mediapipe():
    global MP_FACE_MESH
    if MP_FACE_MESH is None:
        MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=roop.globals.max_faces if hasattr(roop.globals, 'max_faces') else 1,  # From new arg
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def clear_face_analyser() -> None:
    global FACE_ANALYSER
    FACE_ANALYSER = None

def get_one_face(frame: Frame, position: int = 0, occlusion_threshold: float = 0.6, side_adjust: bool = False, face_tracking: bool = False) -> Optional[Face]:
    many_faces = get_many_faces(frame, occlusion_threshold, side_adjust, face_tracking)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def get_many_faces(frame: Frame, occlusion_threshold: float = 0.6, side_adjust: bool = False, face_tracking: bool = False) -> Optional[List[Face]]:
    try:
        initialize_mediapipe()
        faces = get_face_analyser().get(frame)
        if not faces:
            return None
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = MP_FACE_MESH.process(rgb)
        
        if mp_results.multi_face_landmarks:
            for i, landmarks in enumerate(mp_results.multi_face_landmarks):
                if i >= len(faces):
                    break
                lm_list = landmarks.landmark
                h, w = frame.shape[:2]
                points_3d = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lm_list])
                
                # Occlusion check (hands/mouth)
                lip_indices = [11, 13, 14, 15, 16, 308, 324]
                lip_vis = np.mean([lm.presence for lm in lm_list if lm.presence and lm_indices.index(lm) in lip_indices] or [0])
                if lip_vis < occlusion_threshold:
                    faces[i] = None  # Skip occluded face
                
                # Side pose adjustment
                if side_adjust:
                    nose_z = points_3d[1, 2]
                    chin_z = points_3d[152, 2]
                    yaw = np.arctan2(chin_z - nose_z, 100)
                    if abs(yaw) > 0.5:
                        faces[i].bbox[2] *= np.cos(yaw)  # Adjust bbox
                
                # Face tracking (basic embedding persistence for multi-frames)
                if face_tracking and hasattr(faces[i], 'normed_embedding'):
                    # Store in globals for next frame comparison (implement in core.py for full tracking)
                    pass
        
        faces = [f for f in faces if f is not None]
        return faces
    except ValueError:
        return None

def find_similar_face(frame: Frame, reference_face: Face, occlusion_threshold: float = 0.6, side_adjust: bool = False, face_tracking: bool = False) -> Optional[Face]:
    many_faces = get_many_faces(frame, occlusion_threshold, side_adjust, face_tracking)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = np.sum(np.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < similar_face_distance:
                    return face
    return None
