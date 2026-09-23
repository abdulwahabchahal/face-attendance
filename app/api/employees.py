"""
Employee API Routes
───────────────────
POST /register         Register employee with face images
GET  /employees        List all employees
GET  /employee/{id}    Get single employee
DELETE /employee/{id}  Remove employee and all data
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.face_pipeline import augment_and_embed, decode_image
from app.core.schemas import RegisterResponse, EmployeeOut
from app.db.crud import create_employee, delete_employee, get_employee, list_employees, store_embedding
from app.db.database import get_db
from app.modules.rules import parse_shift_time

router = APIRouter()


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_employee(
    employee_id: str = Form(..., description="Unique ID e.g. EMP-001"),
    name: str = Form(...),
    department: str = Form(default="General"),
    role: str = Form(default="employee"),
    shift_start: str = Form(default="09:00", description="HH:MM format"),
    shift_end: str = Form(default="17:00", description="HH:MM format"),
    images: List[UploadFile] = File(..., description="1-10 face photos"),
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new employee and extract face embeddings from uploaded images.

    - Accepts 1–10 images.
    - Applies augmentation to enrich the embedding set.
    - Stores all embeddings associated with the employee_id.
    """
    if await get_employee(db, employee_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Employee '{employee_id}' already exists. Delete first to re-register.",
        )

    if not images or len(images) > 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide between 1 and 10 face images.",
        )

    try:
        parsed_start = parse_shift_time(shift_start)
        parsed_end = parse_shift_time(shift_end)
        shift_start = parsed_start.strftime("%H:%M")
        shift_end = parsed_end.strftime("%H:%M")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    employee = await create_employee(
        db,
        employee_id=employee_id,
        name=name,
        department=department,
        role=role,
        shift_start=shift_start,
        shift_end=shift_end,
    )

    total_stored = 0
    for upload in images:
        raw = await upload.read()
        try:
            rgb = decode_image(raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Bad image: {e}")

        embeddings = augment_and_embed(rgb, n_augments=5)
        if not embeddings:
            raise HTTPException(
                status_code=422,
                detail=f"No face detected in '{upload.filename}'. Use a clear face photo.",
            )

        for vec in embeddings:
            await store_embedding(db, employee_id=employee_id, vector=vec)
            total_stored += 1

    return RegisterResponse(
        employee_id=employee_id,
        embeddings_stored=total_stored,
        message=f"'{name}' registered successfully with {total_stored} embeddings.",
    )


@router.get("/employees", response_model=List[EmployeeOut])
async def get_all_employees(db: AsyncSession = Depends(get_db)):
    """List all registered employees."""
    return await list_employees(db)


@router.get("/employee/{employee_id}", response_model=EmployeeOut)
async def get_single_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    emp = await get_employee(db, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found.")
    return emp


@router.delete("/employee/{employee_id}", status_code=status.HTTP_200_OK)
async def remove_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    """Delete employee and all their embeddings and attendance records."""
    deleted = await delete_employee(db, employee_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Employee not found.")
    return {"message": f"Employee '{employee_id}' deleted successfully."}
