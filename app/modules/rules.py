"""
rules.py — Shift & Attendance Rules Engine
All attendance business logic lives here. Do NOT scatter across other files.

Rules:
  On Time  → check_in <= shift_start + 10 mins  → status = 'Present'
  Late     → check_in > shift_start + 10 mins   → status = 'Late'
  Half Day → check_in after 11:00 AM            → status = 'Half Day'
  Absent   → No check_in by end of day          → status = 'Absent' (batch)
  Overtime → check_out > shift_end              → total_hours > standard hours
"""

import re
from datetime import datetime, timedelta, time
from typing import Optional

from app.core.config import settings


def parse_shift_time(shift_time_str: str) -> time:
    """Parse a shift time string into a time object.

    Supported formats:
    - HH:MM
    - H:MM AM/PM
    - HH AM/PM
    - HPM / HAM
    """
    if not shift_time_str or not shift_time_str.strip():
        raise ValueError("Shift time must not be empty.")

    normalized = shift_time_str.strip().lower().replace(".", "")
    normalized = re.sub(r"\s+", " ", normalized)

    formats = [
        "%I:%M %p",
        "%I %p",
        "%I:%M%p",
        "%I%p",
        "%H:%M",
        "%H%M",
        "%H",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt).time()
        except ValueError:
            continue

    raise ValueError(f"Invalid shift time format: {shift_time_str}")


def parse_shift_datetime(shift_time_str: str, reference_dt: datetime) -> datetime:
    """Convert a shift time string into a datetime on the given reference date."""
    shift_time = parse_shift_time(shift_time_str)
    return reference_dt.replace(
        hour=shift_time.hour,
        minute=shift_time.minute,
        second=0,
        microsecond=0,
    )


def apply_status_rule(check_in_time: datetime, shift_start_str: str, shift_end_str: str = None) -> str:
    """
    Determine attendance status based on check-in time, shift start, and shift end.

    Args:
        check_in_time: Actual datetime when employee checked in
        shift_start_str: Shift start as "HH:MM" or "H:MM AM/PM" string
        shift_end_str: Shift end as "HH:MM" or "H:MM AM/PM" string

    Returns:
        str: 'Present' | 'Late' | 'Half Day' | 'Absent'
    """
    shift_start_dt = parse_shift_datetime(shift_start_str, check_in_time)

    # Ignore check-ins that are far outside the shift window.
    if shift_end_str:
        shift_end_dt = parse_shift_datetime(shift_end_str, check_in_time)
        if check_in_time >= shift_end_dt:
            return "Ignored"

    early_window_start = shift_start_dt - timedelta(hours=settings.EARLY_ARRIVAL_TOLERANCE_HOURS)
    if check_in_time < early_window_start:
        return "Ignored"

    # Early arrival before shift start is still Present.
    if check_in_time < shift_start_dt:
        return "Present"

    # Half Day rule: after the configured half-day hour.
    half_day_threshold = shift_start_dt.replace(
        hour=settings.HALF_DAY_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    if shift_start_dt.hour >= settings.HALF_DAY_HOUR:
        half_day_threshold = shift_start_dt + timedelta(hours=4)

    if check_in_time >= half_day_threshold:
        return "Half Day"

    # Late rule: more than 10 minutes after shift start.
    diff_minutes = (check_in_time - shift_start_dt).total_seconds() / 60
    if diff_minutes > 10:
        return "Late"

    return "Present"


def calculate_total_hours(check_in: datetime, check_out: datetime) -> float:
    """
    Calculate total hours worked between check-in and check-out.

    Args:
        check_in: Check-in datetime
        check_out: Check-out datetime

    Returns:
        float: Hours worked, rounded to 2 decimal places
    """
    if not check_in or not check_out:
        return 0.0
    delta = check_out - check_in
    hours = delta.total_seconds() / 3600
    return round(hours, 2)


def is_overtime(check_out: datetime, shift_end_str: str) -> bool:
    """
    Returns True if employee checked out after their shift end time.

    Args:
        check_out: Checkout datetime
        shift_end_str: Shift end as "HH:MM" or "H:MM AM/PM" string

    Returns:
        bool: True if overtime
    """
    shift_end_dt = parse_shift_datetime(shift_end_str, check_out)
    return check_out > shift_end_dt


def get_overtime_hours(check_out: datetime, shift_end_str: str) -> float:
    """
    Calculate overtime hours worked beyond shift end.

    Returns:
        float: Overtime hours, 0.0 if not overtime
    """
    shift_end_dt = parse_shift_datetime(shift_end_str, check_out)
    if check_out <= shift_end_dt:
        return 0.0
    delta = check_out - shift_end_dt
    return round(delta.total_seconds() / 3600, 2)


def get_standard_hours(shift_start_str: str, shift_end_str: str) -> float:
    """
    Calculate standard shift duration in hours.

    Returns:
        float: Expected hours e.g. 8.0 for 09:00 - 17:00
    """
    start_time = parse_shift_time(shift_start_str)
    end_time = parse_shift_time(shift_end_str)
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    return round((end_minutes - start_minutes) / 60, 2)
