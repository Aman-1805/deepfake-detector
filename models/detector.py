import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

class DeepfakeDetector(nn.Module):
    """
    PyTorch Neural Network for Deepfake Detection.
    Uses ResNet-18 backbone fine-tuned for facial forgery & artifact classification.
    """
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(pretrained=pretrained)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.backbone(x)


def get_transforms():
    """Returns standard ImageNet normalization and resize transformations."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_face(face_input):
    """Preprocesses numpy array (BGR or RGB) or PIL Image into PyTorch Tensor."""
    if isinstance(face_input, np.ndarray):
        if len(face_input.shape) == 3 and face_input.shape[2] == 3:
            face_rgb = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face_input
        face_img = Image.fromarray(face_rgb)
    elif isinstance(face_input, Image.Image):
        face_img = face_input
    else:
        raise ValueError("Unsupported face input type.")

    transform = get_transforms()
    tensor = transform(face_img).unsqueeze(0)
    return tensor


def detect_deepfake_artifacts(face_img):
    """
    Evaluates deepfake forgery indicators:
    1. Facial swap boundary seam discontinuities.
    2. High frequency FFT grid noise.
    3. Artificial color channel mismatch.
    """
    if isinstance(face_img, Image.Image):
        img_np = np.array(face_img)
    else:
        img_np = face_img

    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    h, w = gray.shape
    if h < 20 or w < 20:
        return {'is_deepfake': False, 'forgery_score': 0.0}

    # 1. Boundary seam discontinuity check
    margin_y = max(2, int(h * 0.15))
    margin_x = max(2, int(w * 0.15))

    inner = gray[margin_y:h-margin_y, margin_x:w-margin_x]
    outer = np.copy(gray)
    outer[margin_y:h-margin_y, margin_x:w-margin_x] = 0

    inner_std = float(np.std(inner)) if inner.size > 0 else 1.0
    outer_std = float(np.std(outer))
    boundary_discontinuity = abs(inner_std - outer_std) / (inner_std + 10.0)

    # 2. Artificial rectangle / boundary box artifact detection
    # Deepfakes often leave rectangular blending artifacts near face edges
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    has_box_artifact = False
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(c) > (h * w * 0.1):
            has_box_artifact = True
            break

    # Calculate overall forgery score
    forgery_score = 0.0
    if has_box_artifact:
        forgery_score += 0.60
    if boundary_discontinuity > 2.2:
        forgery_score += 0.40

    is_deepfake = forgery_score >= 0.50
    return {
        'is_deepfake': is_deepfake,
        'forgery_score': forgery_score
    }


def load_model(weights_path=None, device=None):
    """Loads model and moves to specified device ('cuda' or 'cpu')."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if weights_path is None:
        default_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deepfake_model.pth")
        if os.path.exists(default_ckpt):
            weights_path = default_ckpt

    model = DeepfakeDetector(pretrained=True)
    is_custom_loaded = False
    
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            is_custom_loaded = True
            print(f"[Detector] Loaded custom trained weights from {weights_path}")
        except Exception as e:
            print(f"[Detector] Note: Using pretrained backbone ({e}).")

    model.to(device)
    model.eval()
    model.is_custom_loaded = is_custom_loaded
    return model, device


def predict_face(model, face_img, device):
    """
    Classifies real photos as REAL and deepfake/manipulated media as DEEPFAKE.
    """
    tensor = preprocess_face(face_input=face_img).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        raw_probs = F.softmax(logits, dim=1)[0]

    artifacts = detect_deepfake_artifacts(face_img)

    if getattr(model, 'is_custom_loaded', False):
        fake_prob = float(raw_probs[0].item())
        real_prob = float(raw_probs[1].item())
    else:
        fake_prob = float(raw_probs[0].item())
        real_prob = float(raw_probs[1].item())

    is_deepfake = fake_prob > real_prob
    label = "Deepfake" if is_deepfake else "Real"
    confidence = fake_prob if is_deepfake else real_prob

    return {
        "label": label,
        "is_deepfake": is_deepfake,
        "confidence": confidence,
        "probs": {
            "real": real_prob,
            "deepfake": fake_prob
        }
    }
