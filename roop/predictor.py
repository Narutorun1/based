import mediapipe as mp
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from roop.typing import Frame

mp_face_mesh = mp.solutions.face_mesh
face_mesh = None

def initialize_mediapipe():
    global face_mesh
    if face_mesh is None:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=roop.globals.max_faces if hasattr(roop.globals, 'max_faces') else 1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

def get_landmarks(frame: Frame, refine: bool = True, deform_threshold: float = 0.3, mouth_mask: bool = False) -> np.ndarray:
    initialize_mediapipe()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        # Fallback to InsightFace
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0)
        faces = app.get(frame)
        return faces[0].kps if faces else np.zeros((106, 2))
    
    landmarks = results.multi_face_landmarks[0].landmark  # Single face for simplicity
    h, w = frame.shape[:2]
    points_3d = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])
    
    # Deformation check for tongue/mouth (enhanced with height span)
    mouth_indices = [13, 14, 15, 16, 308, 324]  # Outer/inner lip keypoints
    mouth_width = np.max(points_3d[mouth_indices, 0]) - np.min(points_3d[mouth_indices, 0])
    mouth_height = np.max(points_3d[mouth_indices, 1]) - np.min(points_3d[mouth_indices, 1])
    if (mouth_width / w > deform_threshold) or (mouth_height / h > deform_threshold * 0.5):
        return np.zeros((106, 2))  # Skip if extreme deformation
    
    # Eye closure interpolation (improved with bilateral symmetry)
    left_eye_indices = range(33, 42)  # Left eye
    right_eye_indices = range(263, 272)  # Right eye
    left_eye_vis = np.mean([lm.presence for lm in landmarks[i] for i in left_eye_indices if i < len(landmarks)])
    right_eye_vis = np.mean([lm.presence for lm in landmarks[i] for i in right_eye_indices if i < len(landmarks)])
    eye_vis = min(left_eye_vis, right_eye_vis)
    if eye_vis < 0.7:
        # Adjust both eyes with dynamic offset based on visibility
        offset = max(5, int((1 - eye_vis) * 10))  # Increase offset with lower visibility
        for i in left_eye_indices + right_eye_indices:
            if i < len(landmarks):
                points_3d[i, 1] -= offset  # Move eyelids down
    
    # Mouth mask (enhanced with better region definition)
    if mouth_mask:
        mouth_region = np.array([points_3d[i] for i in range(11, 19) if i < len(landmarks)])  # Lips contour
        if len(mouth_region) > 2:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [mouth_region.astype(int)], 255)
            frame = cv2.bitwise_and(frame, frame, mask=~mask)  # Preserve original mouth
    
    # Side pose adjustment (using yaw from depth)
    if roop.globals.side_pose_adjust and hasattr(roop.globals, 'side_pose_adjust'):
        nose_z = points_3d[1, 2]  # Nose tip
        chin_z = points_3d[152, 2]  # Chin
        yaw = np.arctan2(chin_z - nose_z, 100)  # Rough yaw estimate
        if abs(yaw) > 0.5:  # Significant side angle
            # Adjust landmark projection (simple scaling for now)
            points_3d[:, 0] *= np.cos(yaw)
    
    # Map 468 points to 106 for Roop compatibility (improved mapping)
    def map_to_106(points):
        # Approximate mapping (customize based on Roop's 106-point model)
        mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Forehead
                   13, 14, 15, 16, 17, 18,  # Left eyebrow
                   23, 24, 25, 26, 27, 28,  # Right eyebrow
                   33, 34, 35, 36, 37, 38, 39, 40, 41,  # Left eye
                   263, 264, 265, 266, 267, 268, 269, 270, 271,  # Right eye
                   61, 76, 77, 82, 83, 84, 87, 88, 91, 95,  # Nose
                   11, 12, 13, 14, 15, 16, 17, 18,  # Mouth outer
                   308, 310, 311, 312, 317]  # Mouth inner (partial)
        mapped_points = np.zeros((106, 2))
        for i, idx in enumerate(mapping):
            if idx < len(points):
                mapped_points[i] = points[idx, :2]
        return mapped_points
    
    return map_to_106(points_3d)

# Call in swap logic: landmarks = get_landmarks(frame, roop.globals.refine_landmarks, roop.globals.deform_threshold, roop.globals.mouth_mask)
