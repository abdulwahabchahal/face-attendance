"""
Database CRUD helpers — full PRD-compliant operations.
"""

import json
import os
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import EmployeeModel, EmbeddingModel, AttendanceModel, UnknownFaceModel
from app.core.config import settings


# ── Employees ─────────────────────────────────────────────────────────────────

async def create_employee(
    db: AsyncSession,
    employee_id: str,
    name: str,
    department: str = "General",
    role: str = "employee",
    shift_start: str = "09:00",
    shift_end: str = "17:00",
) -> EmployeeModel:
    employee = EmployeeModel(
        id=employee_id,
        name=name,
        department=department,
        role=role,
        shift_start=shift_start,
        shift_end=shift_end,
    )
    db.add(employee)
    await db.flush()
    return employee


async def get_employee(db: AsyncSession, employee_id: str) -> Optional[EmployeeModel]:
    result = await db.execute(select(EmployeeModel).where(EmployeeModel.id == employee_id))
    return result.scalar_one_or_none()


async def list_employees(db: AsyncSession) -> List[EmployeeModel]:
    result = await db.execute(select(EmployeeModel).order_by(EmployeeModel.name))
    return result.scalars().all()


async def delete_employee(db: AsyncSession, employee_id: str) -> bool:
    """
    Delete employee by loading the ORM object first so that SQLAlchemy
    cascade fires and removes all related embeddings and attendance records.
    Raw SQL delete() bypasses cascade — do NOT use it here.
    """
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(EmployeeModel)
        .options(
            selectinload(EmployeeModel.embeddings),
            selectinload(EmployeeModel.attendance_records),
        )
        .where(EmployeeModel.id == employee_id)
    )
    employee = result.scalar_one_or_none()
    if not employee:
        return False
    await db.delete(employee)
    await db.flush()
    return True


# ── Embeddings ────────────────────────────────────────────────────────────────

async def store_embedding(db: AsyncSession, employee_id: str, vector: List[float]) -> EmbeddingModel:
    emb = EmbeddingModel(employee_id=employee_id, vector=json.dumps(vector))
    db.add(emb)
    await db.flush()
    return emb


async def get_all_embeddings(db: AsyncSession) -> List[Tuple[str, List[float]]]:
    """Return list of (employee_id, vector_list) tuples."""
    result = await db.execute(select(EmbeddingModel))
    rows = result.scalars().all()
    return [(r.employee_id, json.loads(r.vector)) for r in rows]


# ── Attendance ─────────────────────────────────────────────────────────────────

async def get_today_record(db: AsyncSession, employee_id: str) -> Optional[AttendanceModel]:
    """Get today's attendance record for an employee."""
    today = date.today()
    result = await db.execute(
        select(AttendanceModel).where(
            AttendanceModel.employee_id == employee_id,
            AttendanceModel.date == today,
        ).order_by(AttendanceModel.check_in_time.desc()) # Get the most recent one
    )
    return result.scalars().first() # Use .first() to prevent the "Multiple Results" crash


async def is_anti_spam(db: AsyncSession, employee_id: str) -> bool:
    """True if this employee was detected within the last 5 minutes."""
    window = datetime.utcnow() - timedelta(minutes=5)
    result = await db.execute(
        select(AttendanceModel).where(
            AttendanceModel.employee_id == employee_id,
            AttendanceModel.date == date.today(),
            AttendanceModel.check_in_time >= window,
        ).order_by(AttendanceModel.check_in_time.desc())
    )
    record = result.scalars().first() # Use .first() here as well
    if record and record.check_in_time and record.check_in_time >= window:
        return True
    return False

    # Check check-out time too
    record = result.scalar_one_or_none()
    if record and record.check_in_time and record.check_in_time >= window:
        return True
    return False


async def process_attendance(
    db: AsyncSession,
    employee_id: str,
    confidence: float,
) -> dict:
    """
    Main attendance decision logic per PRD section 3.3.
    Returns dict with action taken and record details.
    """
    from app.modules.rules import apply_status_rule, calculate_total_hours

    # Anti-spam: ignore if seen in last 5 minutes
    window_5min = datetime.utcnow() - timedelta(minutes=5)
    today = date.today()

    existing = await get_today_record(db, employee_id)

    # Check last interaction time
    if existing:
        last_time = existing.check_out_time or existing.check_in_time
        if last_time and last_time >= window_5min:
            return {"action": "ignored", "reason": "anti-spam: seen within 5 minutes"}

    employee = await get_employee(db, employee_id)
    shift_start = employee.shift_start if employee else "09:00"
    shift_end = employee.shift_end if employee else "17:00"

    now = datetime.utcnow()

    if not existing:
        # First detection today → CHECK-IN
        status = apply_status_rule(now, shift_start)
        record = AttendanceModel(
            employee_id=employee_id,
            date=today,
            check_in_time=now,
            status=status,
            confidence=confidence,
        )
        db.add(record)
        await db.flush()
        return {
            "action": "check_in",
            "status": status,
            "time": now.isoformat(),
            "employee_id": employee_id,
        }
    else:
        # Record exists → Check if we should CHECK-OUT
        if existing.check_out_time is not None:
            # Already checked out — just update last seen timestamp
            existing.check_out_time = now  # update to latest
            await db.flush()
            return {"action": "ignored", "reason": "already checked out today"}

        # Parse shift_end to decide if it's time to check out
        shift_end_hour, shift_end_min = map(int, shift_end.split(":"))
        shift_end_dt = now.replace(hour=shift_end_hour, minute=shift_end_min, second=0, microsecond=0)

        # If current time is past shift end → CHECK-OUT
        if now >= shift_end_dt:
            total = calculate_total_hours(existing.check_in_time, now)
            existing.check_out_time = now
            existing.total_hours = total
            await db.flush()
            return {
                "action": "check_out",
                "total_hours": total,
                "time": now.isoformat(),
                "employee_id": employee_id,
            }
        else:
            # Within shift, just update confidence if better
            return {"action": "ignored", "reason": "within shift, no checkout yet"}


async def get_attendance(
    db: AsyncSession,
    date_filter: Optional[date] = None,
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
) -> List[AttendanceModel]:
    from sqlalchemy.orm import joinedload
    query = select(AttendanceModel).options(joinedload(AttendanceModel.employee))
    if date_filter:
        query = query.where(AttendanceModel.date == date_filter)
    if employee_id:
        query = query.where(AttendanceModel.employee_id == employee_id)
    if department:
        query = query.join(EmployeeModel).where(EmployeeModel.department == department)
    result = await db.execute(query.order_by(AttendanceModel.date.desc(), AttendanceModel.check_in_time.desc()))
    return result.scalars().unique().all()


async def mark_absent_for_today(db: AsyncSession) -> int:
    """
    Mark employees as Absent only if:
    1. They have no check-in record for today
    2. Their own shift has already ended (current time > their shift_end)
    Returns count of employees marked Absent.
    """
    today = date.today()
    now = datetime.now()
    employees = await list_employees(db)
    count = 0
    for emp in employees:
        shift_end_hour, shift_end_min = map(int, emp.shift_end.split(":"))
        shift_end_dt = now.replace(hour=shift_end_hour, minute=shift_end_min, second=0, microsecond=0)
        if now < shift_end_dt:
            continue  # shift not over yet for this employee
        record = await get_today_record(db, emp.id)
        if not record:
            db.add(AttendanceModel(employee_id=emp.id, date=today, status="Absent"))
            count += 1
    await db.flush()
    return count


async def get_today_stats(db: AsyncSession) -> dict:
    """Return today's dashboard stats."""
    today = date.today()
    result = await db.execute(
        select(AttendanceModel).where(AttendanceModel.date == today)
    )
    records = result.scalars().all()
    total_employees_result = await db.execute(select(func.count(EmployeeModel.id)))
    total_employees = total_employees_result.scalar() or 0

    present = sum(1 for r in records if r.status in ("Present", "Late", "Half Day", "Early Leave") and r.check_in_time)
    late = sum(1 for r in records if r.status == "Late")
    half_day = sum(1 for r in records if r.status == "Half Day")
    absent = sum(1 for r in records if r.status == "Absent")
    early_leave = sum(1 for r in records if r.status == "Early Leave")

    return {
        "total_employees": total_employees,
        "present": present,
        "late": late,
        "half_day": half_day,
        "absent": absent,
        "early_leave": early_leave,
    }


async def mark_early_leave(db: AsyncSession, employee_id: str) -> dict:
    """
    Mark an employee as Early Leave.
    - Must have checked in today
    - Must not have already checked out
    - Records check_out_time = now, calculates total_hours, sets status = Early Leave
    """
    from app.modules.rules import calculate_total_hours

    record = await get_today_record(db, employee_id)
    if not record:
        return {"success": False, "reason": "No check-in found for today"}
    if record.check_out_time is not None:
        return {"success": False, "reason": "Already checked out today"}
    if not record.check_in_time:
        return {"success": False, "reason": "No check-in time recorded"}

    now = datetime.now()
    total = calculate_total_hours(record.check_in_time, now)
    record.check_out_time = now
    record.total_hours = total
    record.status = "Early Leave"
    await db.flush()
    return {
        "success": True,
        "employee_id": employee_id,
        "check_out_time": now.isoformat(),
        "total_hours": total,
        "status": "Early Leave",
    }


# ── Unknown Faces ─────────────────────────────────────────────────────────────

async def log_unknown_face(
    db: AsyncSession,
    snapshot_path: Optional[str] = None,
) -> UnknownFaceModel:
    """
    Log an unknown face to the database.
    Deduplication is handled upstream in recognition.py using
    in-memory embedding similarity — do not duplicate that logic here.
    """
    record = UnknownFaceModel(snapshot_path=snapshot_path)
    db.add(record)
    await db.flush()
    return record


async def get_unknown_faces(db: AsyncSession, unresolved_only: bool = False) -> List[UnknownFaceModel]:
    query = select(UnknownFaceModel)
    if unresolved_only:
        query = query.where(UnknownFaceModel.resolved == False)
    result = await db.execute(query.order_by(UnknownFaceModel.timestamp.desc()))
    return result.scalars().all()


async def resolve_unknown_face(db: AsyncSession, face_id: int) -> bool:
    result = await db.execute(select(UnknownFaceModel).where(UnknownFaceModel.id == face_id))
    record = result.scalar_one_or_none()
    if not record:
        return False
    record.resolved = True
    await db.flush()
    return True