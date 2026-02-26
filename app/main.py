"""
Face Recognition Attendance System — Main FastAPI entry point

Includes APScheduler job that runs every 30 minutes and automatically
marks absent any employee whose shift has ended with no check-in.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.api import employees, recognition, attendance
from app.db.database import init_db, AsyncSessionLocal
from app.db.crud import mark_absent_for_today
from app.core.config import settings

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()


async def run_absent_job():
    """
    Scheduler job: checks every 30 minutes.
    For each employee whose shift_end has passed today,
    writes an Absent record if they never checked in.
    """
    try:
        async with AsyncSessionLocal() as db:
            count = await mark_absent_for_today(db)
            await db.commit()
            if count > 0:
                logger.info(f"[Absent Scheduler] Marked {count} employee(s) as Absent.")
    except Exception as e:
        logger.error(f"[Absent Scheduler] Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, create dirs, start scheduler."""
    await init_db()
    os.makedirs(settings.UNKNOWN_FACES_DIR, exist_ok=True)

    # Run every 30 minutes — catches employees as their shifts end throughout the day
    scheduler.add_job(
        run_absent_job,
        trigger=IntervalTrigger(minutes=30),
        id="absent_marker",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Absent scheduler started — runs every 30 minutes.")

    yield

    scheduler.shutdown()
    logger.info("Absent scheduler stopped.")


app = FastAPI(
    title="Face Recognition Attendance System",
    description="AI-powered employee attendance tracking via facial recognition.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(employees.router, prefix="/api/v1", tags=["Employees"])
app.include_router(recognition.router, prefix="/api/v1", tags=["Recognition"])
app.include_router(attendance.router, prefix="/api/v1", tags=["Attendance"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}


# Serve unknown face snapshots so browser can display them as images
app.mount("/unknown_faces", StaticFiles(directory=settings.UNKNOWN_FACES_DIR), name="unknown_faces")

# Serve frontend — MUST be last
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")