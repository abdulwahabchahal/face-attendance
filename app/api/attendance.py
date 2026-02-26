"""
Attendance API Routes
─────────────────────
GET  /attendance                   Query attendance records (filterable)
GET  /attendance/stats             Today's dashboard stats
GET  /attendance/export            Download attendance CSV
POST /attendance/early-leave       Mark employee as early leave
POST /attendance/mark-absent       Run absent marking batch job
GET  /unknown-faces                List unknown face alerts
POST /unknown-faces/{id}/resolve   Mark unknown face as resolved
"""

import csv
import io
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import (
    AttendanceRecord, DashboardStats, UnknownFaceOut,
    EarlyLeaveRequest, EarlyLeaveResponse,
)
from app.db.crud import (
    get_attendance,
    get_today_stats,
    get_unknown_faces,
    mark_absent_for_today,
    mark_early_leave,
    resolve_unknown_face,
)
from app.db.database import get_db

router = APIRouter()


@router.get("/attendance", response_model=List[AttendanceRecord])
async def query_attendance(
    date_filter: Optional[date] = Query(None, description="Filter by date YYYY-MM-DD"),
    employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
    department: Optional[str] = Query(None, description="Filter by department"),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve attendance records with optional filters."""
    records = await get_attendance(
        db,
        date_filter=date_filter,
        employee_id=employee_id,
        department=department,
    )
    result = []
    for r in records:
        result.append(AttendanceRecord(
            id=r.id,
            employee_id=r.employee_id,
            employee_name=r.employee.name if r.employee else None,
            date=r.date,
            check_in_time=r.check_in_time,
            check_out_time=r.check_out_time,
            total_hours=r.total_hours,
            status=r.status,
            confidence=r.confidence,
        ))
    return result


@router.get("/attendance/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """Return today's attendance summary for the dashboard."""
    stats = await get_today_stats(db)
    return DashboardStats(**stats)


@router.get("/attendance/export")
async def export_attendance_csv(
    date_filter: Optional[date] = Query(None, description="Filter by date"),
    department: Optional[str] = Query(None, description="Filter by department"),
    db: AsyncSession = Depends(get_db),
):
    """Download attendance records as a CSV file with day and time columns."""
    records = await get_attendance(db, date_filter=date_filter, department=department)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Employee ID", "Name",
        "Day", "Date",
        "Check In Time", "Check Out Time",
        "Total Hours", "Status"
    ])

    for r in records:
        # Get day name e.g. Monday, Tuesday
        day_name = r.date.strftime("%A") if r.date else ""
        writer.writerow([
            r.id,
            r.employee_id,
            r.employee.name if r.employee else "",
            day_name,
            r.date.strftime("%d-%m-%Y") if r.date else "",
            r.check_in_time.strftime("%I:%M:%S %p") if r.check_in_time else "",
            r.check_out_time.strftime("%I:%M:%S %p") if r.check_out_time else "",
            r.total_hours or "",
            r.status,
        ])

    output.seek(0)
    filename = f"attendance_{date_filter or date.today()}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/attendance/early-leave", response_model=EarlyLeaveResponse)
async def early_leave(
    request: EarlyLeaveRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an employee as Early Leave.
    Records their checkout time now and sets status to 'Early Leave'.
    Employee must have checked in today and not yet checked out.
    """
    result = await mark_early_leave(db, employee_id=request.employee_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["reason"])
    return EarlyLeaveResponse(**result)


@router.post("/attendance/mark-absent")
async def run_absent_batch(db: AsyncSession = Depends(get_db)):
    """Mark all employees with no check-in today as Absent (runs automatically via scheduler)."""
    count = await mark_absent_for_today(db)
    return {"message": f"Marked {count} employees as Absent for today."}


@router.get("/unknown-faces", response_model=List[UnknownFaceOut])
async def list_unknown_faces(
    unresolved_only: bool = Query(False, description="Show only unresolved alerts"),
    db: AsyncSession = Depends(get_db),
):
    """List unknown face alerts captured by the camera."""
    return await get_unknown_faces(db, unresolved_only=unresolved_only)


@router.post("/unknown-faces/{face_id}/resolve")
async def resolve_face_alert(face_id: int, db: AsyncSession = Depends(get_db)):
    """Mark an unknown face alert as resolved."""
    success = await resolve_unknown_face(db, face_id)
    if not success:
        raise HTTPException(status_code=404, detail="Unknown face record not found.")
    return {"message": f"Face alert {face_id} resolved."}