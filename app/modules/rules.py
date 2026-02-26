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

from datetime import datetime, timedelta
from typing import Optional


def apply_status_rule(check_in_time: datetime, shift_start_str: str) -> str:
    """
    Determine attendance status based on check-in time and shift start.

    Args:
        check_in_time: Actual datetime when employee checked in
        shift_start_str: Shift start as "HH:MM" string e.g. "09:00"

    Returns:
        str: 'Present' | 'Late' | 'Half Day'
    """
    # Parse shift_start into today's datetime
    hour, minute = map(int, shift_start_str.split(":"))
    shift_start_dt = check_in_time.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # Half Day rule: check-in at or after 11:00 AM
    if check_in_time.hour >= 11:
        return "Half Day"

    # Late rule: more than 10 minutes after shift start
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
        shift_end_str: Shift end as "HH:MM" string e.g. "17:00"

    Returns:
        bool: True if overtime
    """
    hour, minute = map(int, shift_end_str.split(":"))
    shift_end_dt = check_out.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return check_out > shift_end_dt


def get_overtime_hours(check_out: datetime, shift_end_str: str) -> float:
    """
    Calculate overtime hours worked beyond shift end.

    Returns:
        float: Overtime hours, 0.0 if not overtime
    """
    if not is_overtime(check_out, shift_end_str):
        return 0.0
    hour, minute = map(int, shift_end_str.split(":"))
    shift_end_dt = check_out.replace(hour=hour, minute=minute, second=0, microsecond=0)
    delta = check_out - shift_end_dt
    return round(delta.total_seconds() / 3600, 2)


def get_standard_hours(shift_start_str: str, shift_end_str: str) -> float:
    """
    Calculate standard shift duration in hours.

    Returns:
        float: Expected hours e.g. 8.0 for 09:00 - 17:00
    """
    start_h, start_m = map(int, shift_start_str.split(":"))
    end_h, end_m = map(int, shift_end_str.split(":"))
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    return round((end_minutes - start_minutes) / 60, 2)
