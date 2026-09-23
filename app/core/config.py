"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./attendance.db"

    # ── Model ────────────────────────────────────────────
    MODEL_NAME: str = "vggface2"
    EMBEDDING_DIM: int = 512

    # ── Recognition ──────────────────────────────────────
    RECOGNITION_THRESHOLD: float = 0.65
    CONFIRM_FRAMES: int = 3

    # ── Attendance Rules ──────────────────────────────────
    ANTI_SPAM_MINUTES: int = 5          # ignore same employee within N minutes
    DEFAULT_SHIFT_START: str = "09:00"
    DEFAULT_SHIFT_END: str = "17:00"
    LATE_GRACE_MINUTES: int = 10        # minutes after shift start before marking Late
    HALF_DAY_HOUR: int = 11             # hour at or after which check-in = Half Day
    EARLY_ARRIVAL_TOLERANCE_HOURS: int = 4  # ignore check-ins more than N hours before shift start

    # ── Paths ─────────────────────────────────────────────
    UNKNOWN_FACES_DIR: str = "unknown_faces"

    CAMERA_INDEX: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
