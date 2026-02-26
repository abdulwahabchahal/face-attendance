"""
Database layer: async SQLAlchemy engine, session factory, and ORM models.
Updated to support full PRD requirements:
 - employees with shift times and department
 - attendance with check-in/check-out/total_hours/status
 - unknown_faces table
"""

from datetime import datetime, date
from sqlalchemy import Column, String, Float, DateTime, Date, Time, Text, Integer, ForeignKey, Boolean
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class EmployeeModel(Base):
    __tablename__ = "employees"

    id = Column(String, primary_key=True)               # e.g. "EMP-001"
    name = Column(String, nullable=False)
    department = Column(String, default="General")
    role = Column(String, default="employee")           # "employee" | "admin"
    shift_start = Column(String, default="09:00")       # stored as "HH:MM" string
    shift_end = Column(String, default="17:00")
    status = Column(String, default="active")           # "active" | "inactive"
    created_at = Column(DateTime, default=datetime.utcnow)

    embeddings = relationship("EmbeddingModel", back_populates="employee", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceModel", back_populates="employee", cascade="all, delete-orphan")


class EmbeddingModel(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    vector = Column(Text, nullable=False)               # JSON-serialised list[float]
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("EmployeeModel", back_populates="embeddings")


class AttendanceModel(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String, ForeignKey("employees.id"), nullable=False)
    date = Column(Date, nullable=False)
    check_in_time = Column(DateTime, nullable=True)
    check_out_time = Column(DateTime, nullable=True)
    total_hours = Column(Float, nullable=True)
    status = Column(String, default="Present")          # Present | Late | Half Day | Absent
    confidence = Column(Float, nullable=True)

    employee = relationship("EmployeeModel", back_populates="attendance_records")


class UnknownFaceModel(Base):
    __tablename__ = "unknown_faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    snapshot_path = Column(String, nullable=True)
    resolved = Column(Boolean, default=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
