# Face Recognition Attendance System v2.0
**PRD-compliant** — Auto check-in/check-out, shift rules, unknown face logging, CSV export

---

## Quick Start

```bash
# 1. Clone / unzip project
cd face_attendance

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env

# 5. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Open browser
# http://localhost:8000
```

---

## What's New in v2.0 (PRD Changes)

| Feature | v1 | v2 |
|---|---|---|
| Check-in only | ✅ | ✅ |
| Check-out auto-detection | ❌ | ✅ |
| Shift rules (Late/Half Day) | ❌ | ✅ |
| Anti-spam (5 min window) | ❌ | ✅ |
| Unknown face snapshot | ❌ | ✅ |
| Department field | ❌ | ✅ |
| CSV export | ❌ | ✅ |
| Dashboard stats | ❌ | ✅ |
| Absent marking batch | ❌ | ✅ |

---

## Folder Structure

```
face_attendance/
├── app/
│   ├── main.py                  ← FastAPI entry point
│   ├── api/
│   │   ├── employees.py         ← Register/list/delete employees
│   │   ├── recognition.py       ← Face recognition + check-in/out
│   │   └── attendance.py        ← Reports, CSV export, alerts
│   ├── core/
│   │   ├── config.py            ← All settings (env vars)
│   │   ├── face_pipeline.py     ← MTCNN + FaceNet + cosine matching
│   │   └── schemas.py           ← Pydantic models
│   ├── db/
│   │   ├── database.py          ← SQLAlchemy ORM models
│   │   └── crud.py              ← All DB operations
│   ├── modules/
│   │   └── rules.py             ← Shift rules engine (Late/Half Day/Overtime)
│   └── static/
│       └── index.html           ← Full dashboard UI
├── unknown_faces/               ← Snapshots of unrecognised persons
├── attendance.db                ← SQLite database (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Endpoints

### Employees
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/register | Register employee with face images |
| GET | /api/v1/employees | List all employees |
| GET | /api/v1/employee/{id} | Get single employee |
| DELETE | /api/v1/employee/{id} | Delete employee + data |

### Recognition
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/recognize | Identify face + auto check-in/out |
| POST | /api/v1/recognize_video | Process video frame (multi-face) |

### Attendance
| Method | Endpoint | Description |
|---|---|---|
| GET | /api/v1/attendance | Query records (filter by date/dept/employee) |
| GET | /api/v1/attendance/stats | Today's dashboard summary |
| GET | /api/v1/attendance/export | Download CSV |
| POST | /api/v1/attendance/mark-absent | Run absent batch (end of day) |
| GET | /api/v1/unknown-faces | List unknown face alerts |
| POST | /api/v1/unknown-faces/{id}/resolve | Resolve an alert |

Full interactive docs: http://localhost:8000/docs

---

## Shift & Attendance Rules

All rules are in `app/modules/rules.py`.

| Rule | Condition | Status |
|---|---|---|
| On Time | check_in ≤ shift_start + 10 min | Present |
| Late | check_in > shift_start + 10 min | Late |
| Half Day | check_in at or after 11:00 AM | Half Day |
| Absent | No check_in by end of day | Absent (batch job) |
| Overtime | check_out > shift_end | total_hours > standard |
| Anti-Spam | Same employee within 5 minutes | Ignored |

---

## Common Errors

| Error | Fix |
|---|---|
| `dlib build fails` | `pip install cmake` first, then retry |
| `Camera not found` | Try `CAMERA_INDEX=1` in .env |
| `No module named cv2` | `pip install opencv-python-headless` |
| `Database locked` | Restart the app |
| `No face detected` | Use a clear, well-lit frontal photo |
