"""
Pydantic schemas for request validation and response serialisation.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import List, Optional, Dict

from pydantic import BaseModel, Field


# ── Employees ─────────────────────────────────────────────────────────────────

class EmployeeOut(BaseModel):
    id: str
    name: str
    department: str
    role: str
    shift_start: str
    shift_end: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class RegisterResponse(BaseModel):
    employee_id: str
    embeddings_stored: int
    message: str


# ── Recognition ──────────────────────────────────────────────────────────────

class RecognitionResult(BaseModel):
    employee_id: str
    name: Optional[str] = None
    confidence: float
    status: str = Field(..., description="'recognised' | 'unknown'")
    action: Optional[str] = None        # check_in | check_out | ignored
    reason: Optional[str] = None


class FaceDetection(BaseModel):
    box: Dict[str, int]
    employee_id: str
    name: Optional[str] = None
    confidence: float
    status: str
    action: Optional[str] = None
    reason: Optional[str] = None


class VideoRecognitionResult(BaseModel):
    detections: List[FaceDetection]
    total_faces: int


# ── Attendance ────────────────────────────────────────────────────────────────

class AttendanceRecord(BaseModel):
    id: int
    employee_id: str
    employee_name: Optional[str] = None
    date: date
    check_in_time: Optional[datetime] = None
    check_out_time: Optional[datetime] = None
    total_hours: Optional[float] = None
    status: str
    confidence: Optional[float] = None

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_employees: int
    present: int
    late: int
    half_day: int
    absent: int
    early_leave: int = 0


class EarlyLeaveRequest(BaseModel):
    employee_id: str


class EarlyLeaveResponse(BaseModel):
    success: bool
    reason: Optional[str] = None
    employee_id: Optional[str] = None
    check_out_time: Optional[str] = None
    total_hours: Optional[float] = None
    status: Optional[str] = None


# ── Unknown Faces ──────────────────────────────────────────────────────────────

class UnknownFaceOut(BaseModel):
    id: int
    timestamp: datetime
    snapshot_path: Optional[str] = None
    resolved: bool

    class Config:
        from_attributes = True