"""
Core ML pipeline
────────────────
face_detector  → MTCNN
embedder       → InceptionResnetV1 (pretrained on VGGFace2)
matcher        → cosine similarity against stored embeddings
augmenter      → imgaug augmentation pipeline for registration

All heavy objects are singletons created once at import time.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from imgaug import augmenters as iaa
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ── Singletons ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_detector() -> MTCNN:
    return MTCNN(keep_all=True, device=DEVICE, min_face_size=40)


@lru_cache(maxsize=1)
def get_embedder() -> InceptionResnetV1:
    return InceptionResnetV1(pretrained=settings.MODEL_NAME).eval().to(DEVICE)


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def build_augmenter() -> iaa.Sequential:
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20), shear=(-5, 5)),
        iaa.Multiply((0.75, 1.25)),
        iaa.AdditiveGaussianNoise(scale=(5, 15)),
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.LinearContrast((0.8, 1.2)),
        iaa.Sharpen(alpha=(0, 0.8)),
        iaa.CoarseDropout(0.02, size_percent=0.1),
    ], random_order=False)


# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG / PNG) → RGB numpy array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Face detection ────────────────────────────────────────────────────────────

def detect_faces(rgb_image: np.ndarray) -> Optional[torch.Tensor]:
    """Run MTCNN on an RGB image. Returns face tensor(s) or None."""
    detector = get_detector()
    faces = detector(Image.fromarray(rgb_image))
    return faces


def detect_faces_with_boxes(rgb_image: np.ndarray):
    """Returns (boxes, probs) from MTCNN.detect for drawing bounding boxes."""
    detector = get_detector()
    boxes, probs = detector.detect(Image.fromarray(rgb_image))
    return boxes, probs


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(face_tensor: torch.Tensor) -> List[float]:
    """Compute a 512-D L2-normalised embedding for a single aligned face tensor."""
    embedder = get_embedder()
    if face_tensor.dim() == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(DEVICE)
    embedding = embedder(face_tensor)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.squeeze().cpu().tolist()


def extract_embeddings_from_image(rgb_image: np.ndarray) -> List[List[float]]:
    """Detect all faces in an image and return their embeddings."""
    faces = detect_faces(rgb_image)
    if faces is None:
        return []
    if isinstance(faces, torch.Tensor):
        faces = [faces] if faces.dim() == 3 else list(faces)
    return [extract_embedding(f) for f in faces]


# ── Augmentation for registration ─────────────────────────────────────────────

def augment_and_embed(rgb_image: np.ndarray, n_augments: int = 8) -> List[List[float]]:
    """
    Detect the face in rgb_image, produce n_augments augmented variants,
    and return embeddings for all of them (including the original).
    """
    augmenter = build_augmenter()
    all_embeddings: List[List[float]] = []

    orig_embeddings = extract_embeddings_from_image(rgb_image)
    all_embeddings.extend(orig_embeddings)

    if not orig_embeddings:
        return []

    augmented_images = augmenter.augment_images([rgb_image] * n_augments)
    for aug_img in augmented_images:
        aug_embeddings = extract_embeddings_from_image(aug_img)
        all_embeddings.extend(aug_embeddings)

    return all_embeddings


# ── Cosine similarity matching ────────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def match_embedding(
    query: List[float],
    stored: List[Tuple[str, List[float]]],
    threshold: float = None,
) -> Tuple[str, float]:
    """
    Compare query embedding against all stored (employee_id, vector) pairs.
    Returns (employee_id, best_score) or ("unknown", best_score).
    """
    if threshold is None:
        threshold = settings.RECOGNITION_THRESHOLD

    if not stored:
        return "unknown", 0.0

    best_id = "unknown"
    best_score = -1.0

    for employee_id, vector in stored:
        score = cosine_similarity(query, vector)
        if score > best_score:
            best_score = score
            best_id = employee_id

    if best_score < threshold:
        return "unknown", best_score

    return best_id, best_score


def save_unknown_snapshot(rgb_image: np.ndarray, face_box, output_dir: str) -> Optional[str]:
    """
    Crop the unknown face and save it as a snapshot image.
    Returns the saved file path or None.
    """
    import os
    from datetime import datetime

    try:
        os.makedirs(output_dir, exist_ok=True)
        x1, y1, x2, y2 = [int(c) for c in face_box]
        # Add padding
        pad = 20
        h, w = rgb_image.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = rgb_image[y1:y2, x1:x2]
        bgr_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        # encode face centre into filename for per-region deduplication
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        filename = f"unknown_{ts}_cx{cx}cy{cy}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, bgr_crop)
        return filepath
    except Exception as e:
        logger.error(f"Failed to save unknown face snapshot: {e}")
        return None