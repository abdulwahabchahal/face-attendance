"""
/recognize_video — POST  Real-time video frame processing with multiple face detection
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
import numpy as np

from app.core.config import settings
from app.core.face_pipeline import decode_image, detect_faces_with_boxes, extract_embedding, match_embedding
from app.core.schemas import VideoRecognitionResult, FaceDetection
from app.db.crud import get_all_embeddings, get_user, is_duplicate, log_attendance
from app.db.database import get_db

router = APIRouter()


@router.post("/recognize_video", response_model=VideoRecognitionResult)
async def recognize_video_frame(
    image: UploadFile = File(..., description="Video frame containing faces"),
    lecture_id: Optional[str] = Form(None, description="Lecture ID for attendance logging"),
    threshold: Optional[float] = Form(None, description="Override default recognition threshold"),
    db: AsyncSession = Depends(get_db),
):
    """
    Process a video frame and detect/recognize multiple faces.
    
    Returns:
    - List of detected faces with bounding boxes
    - Recognition results for each face
    - Attendance logging status
    """
    raw = await image.read()
    try:
        rgb_image = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    # Detect all faces with bounding boxes
    boxes, probs = detect_faces_with_boxes(rgb_image)
    
    if boxes is None or len(boxes) == 0:
        return VideoRecognitionResult(
            detections=[],
            total_faces=0
        )

    # Load all stored embeddings once
    stored = await get_all_embeddings(db)
    effective_threshold = threshold if threshold is not None else settings.RECOGNITION_THRESHOLD
    
    detections: List[FaceDetection] = []
    
    # Process each detected face
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Extract face crop
        face_crop = rgb_image[y1:y2, x1:x2]
        
        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue
        
        try:
            # Get embedding for this face
            from facenet_pytorch import MTCNN
            from app.core.face_pipeline import get_detector, get_embedder
            
            mtcnn = get_detector()
            face_tensor = mtcnn(face_crop)
            
            if face_tensor is not None:
                embedding = extract_embedding(face_tensor)
                
                # Match against database
                if stored:
                    user_id, confidence = match_embedding(embedding, stored, threshold=effective_threshold)
                else:
                    user_id, confidence = "unknown", 0.0
                
                recognised = user_id != "unknown"
                user_name = None
                attendance_logged = False
                
                if recognised:
                    user = await get_user(db, user_id)
                    user_name = user.name if user else user_id
                    
                    # Log attendance if requested
                    if lecture_id:
                        duplicate = await is_duplicate(db, user_id=user_id, lecture_id=lecture_id)
                        if not duplicate:
                            await log_attendance(db, user_id=user_id, lecture_id=lecture_id, confidence=confidence)
                            attendance_logged = True
                
                detections.append(FaceDetection(
                    box={"x": x1, "y": y1, "width": x2-x1, "height": y2-y1},
                    user_id=user_id,
                    name=user_name,
                    confidence=round(confidence, 4),
                    status="recognised" if recognised else "unknown",
                    attendance_logged=attendance_logged
                ))
        
        except Exception as e:
            # If extraction fails for this face, skip it
            print(f"Error processing face {i}: {e}")
            continue
    
    return VideoRecognitionResult(
        detections=detections,
        total_faces=len(detections)
    )
