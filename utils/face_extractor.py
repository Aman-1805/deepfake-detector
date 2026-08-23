import cv2
import numpy as np
from PIL import Image
import functools

@functools.lru_cache(maxsize=1)
def get_cascade():
    """Loads OpenCV frontal face Haar Cascade (cached in memory)."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def extract_faces(image_input, max_faces=5, padding=0.2):
    """
    Detects and crops faces from image (numpy array or PIL Image).
    Returns list of dicts: [{'face_crop': np.ndarray (BGR), 'box': (x, y, w, h)}]
    """
    if isinstance(image_input, Image.Image):
        # Convert PIL Image to OpenCV BGR format
        img_np = np.array(image_input)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np
    else:
        img_bgr = image_input

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = get_cascade()
    
    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    results = []
    h_img, w_img = img_bgr.shape[:2]

    # Sort faces by size (largest first)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]

    for (x, y, w, h) in faces:
        # Add padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        face_crop = img_bgr[y1:y2, x1:x2]
        results.append({
            'face_crop': face_crop,
            'box': (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        })

    return results

def extract_primary_face(image_input, fallback_to_full=True):
    """
    Extracts the largest face from image. If no face found and fallback_to_full is True,
    returns the entire image marked as full frame fallback.
    """
    faces = extract_faces(image_input, max_faces=1)
    if faces:
        return faces[0]['face_crop'], faces[0]['box'], True
    
    if fallback_to_full:
        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if len(img_np.shape) == 3 else img_np
        else:
            img_bgr = image_input
        h, w = img_bgr.shape[:2]
        return img_bgr, (0, 0, w, h), False

    return None, None, False
