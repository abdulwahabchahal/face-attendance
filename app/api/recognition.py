"""
Recognition API Routes
──────────────────────
POST /recognize        Identify a single face image + trigger check-in/out
POST /recognize_video  Process a video frame (multiple faces)

Unknown face deduplication uses an in-memory dict keyed by embedding similarity.
Same unknown face within 2 minutes is logged only once.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime, timedelta
import numpy as np

from app.core.config import settings
from app.core.face_pipeline import (
    decode_image,
    detect_faces_with_boxes,
    extract_embedding,
    extract_embeddings_from_image,
    match_embedding,
    save_unknown_snapshot,
    cosine_similarity,
)
from app.core.schemas import RecognitionResult, VideoRecognitionResult, FaceDetection
from app.db.crud import (
    get_all_embeddings,
    get_employee,
    process_attendance,
    log_unknown_face,
)
from app.db.database import get_db

router = APIRouter()

# ── In-memory unknown face cooldown ──────────────────────────────────────────
# Stores: list of {"embedding": [...], "last_seen": datetime}
# A new unknown is only logged if no existing entry has cosine similarity > 0.75
# within the last 2 minutes.
_unknown_cache: List[dict] = []
UNKNOWN_COOLDOWN_MINUTES = 2
UNKNOWN_SIMILARITY_THRESHOLD = 0.75


def _is_duplicate_unknown(embedding: List[float]) -> bool:
    """
    Returns True if this unknown face was already seen recently.
    Compares embedding against all cached unknowns from last 2 minutes.
    """
    global _unknown_cache
    now = datetime.now()
    cutoff = now - timedelta(minutes=UNKNOWN_COOLDOWN_MINUTES)

    # Remove expired entries
    _unknown_cache = [e for e in _unknown_cache if e["last_seen"] >= cutoff]

    for entry in _unknown_cache:
        sim = cosine_similarity(embedding, entry["embedding"])
        if sim >= UNKNOWN_SIMILARITY_THRESHOLD:
            return True  # same unknown person seen recently
    return False


def _cache_unknown(embedding: List[float]):
    """Add this unknown face embedding to the cache."""
    _unknown_cache.append({
        "embedding": embedding,
        "last_seen": datetime.now(),
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/recognize", response_model=RecognitionResult)
async def recognize_face(
    image: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Identify a face and handle check-in/check-out.
    Unknown faces are deduplicated by embedding similarity in memory.
    """
    raw = await image.read()
    try:
        rgb = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    query_embeddings = extract_embeddings_from_image(rgb)
    if not query_embeddings:
        raise HTTPException(status_code=422, detail="No face detected.")

    query_vec = query_embeddings[0]

    stored = await get_all_embeddings(db)
    if not stored:
        raise HTTPException(status_code=404, detail="No registered employees found.")

    effective_threshold = threshold if threshold is not None else settings.RECOGNITION_THRESHOLD
    employee_id, confidence = match_embedding(query_vec, stored, threshold=effective_threshold)

    recognised = employee_id != "unknown"
    employee_name: Optional[str] = None
    action = None

    if recognised:
        employee = await get_employee(db, employee_id)
        employee_name = employee.name if employee else employee_id
        result = await process_attendance(db, employee_id=employee_id, confidence=confidence)
        action = result.get("action")
        reason = result.get("reason")
    else:
        # Only log if this unknown face hasn't been seen in the last 2 minutes
        if not _is_duplicate_unknown(query_vec):
            _cache_unknown(query_vec)
            boxes, _ = detect_faces_with_boxes(rgb)
            snapshot_path = None
            if boxes is not None and len(boxes) > 0:
                snapshot_path = save_unknown_snapshot(rgb, boxes[0], settings.UNKNOWN_FACES_DIR)
            await log_unknown_face(db, snapshot_path=snapshot_path)

    return RecognitionResult(
        employee_id=employee_id,
        name=employee_name,
        confidence=round(confidence, 4),
        status="recognised" if recognised else "unknown",
        action=action,
        reason=reason if recognised else None,
    )


@router.post("/recognize_video", response_model=VideoRecognitionResult)
async def recognize_video_frame(
    image: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Process a video frame with multiple faces.
    Each distinct unknown face is logged once per 2 minutes.
    Multiple different unknown faces in the same frame are all captured.
    """
    raw = await image.read()
    try:
        rgb_image = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    boxes, probs = detect_faces_with_boxes(rgb_image)
    if boxes is None or len(boxes) == 0:
        return VideoRecognitionResult(detections=[], total_faces=0)

    stored = await get_all_embeddings(db)
    effective_threshold = threshold if threshold is not None else settings.RECOGNITION_THRESHOLD

    from app.core.face_pipeline import get_detector
    mtcnn = get_detector()
    detections: List[FaceDetection] = []

    for i, (box, prob) in enumerate(zip(boxes, probs)):
        x1, y1, x2, y2 = [int(c) for c in box]
        face_crop = rgb_image[max(0, y1):y2, max(0, x1):x2]
        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        try:
            from PIL import Image
            face_tensor = mtcnn(Image.fromarray(face_crop))
            if face_tensor is None:
                continue

            embedding = extract_embedding(face_tensor)

            if stored:
                employee_id, confidence = match_embedding(embedding, stored, threshold=effective_threshold)
            else:
                employee_id, confidence = "unknown", 0.0

            recognised = employee_id != "unknown"
            employee_name = None
            action = None

            if recognised:
                employee = await get_employee(db, employee_id)
                employee_name = employee.name if employee else employee_id
                result = await process_attendance(db, employee_id=employee_id, confidence=confidence)
                action = result.get("action")
                reason = result.get("reason")
            else:
                # Each distinct unknown face gets its own cache entry
                if not _is_duplicate_unknown(embedding):
                    _cache_unknown(embedding)
                    snapshot_path = save_unknown_snapshot(rgb_image, box, settings.UNKNOWN_FACES_DIR)
                    await log_unknown_face(db, snapshot_path=snapshot_path)

            detections.append(FaceDetection(
                box={"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                employee_id=employee_id,
                name=employee_name,
                confidence=round(confidence, 4),
                status="recognised" if recognised else "unknown",
                action=action,
                reason=reason if recognised else None,
            ))

        except Exception as e:
            print(f"Error processing face {i}: {e}")
            continue

    return VideoRecognitionResult(detections=detections, total_faces=len(detections))