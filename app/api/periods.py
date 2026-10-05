"""Period (timetable) CRUD — per-period attendance definitions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Period
from ..schemas import PeriodCreate, PeriodOut, PeriodUpdate
from .deps import get_current_admin, get_db

router = APIRouter(prefix="/periods", tags=["periods"])


def _get_or_404(db: Session, period_id: int) -> Period:
    p = db.get(Period, period_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"period {period_id} not found")
    return p


@router.post("", response_model=PeriodOut, status_code=status.HTTP_201_CREATED, summary="Create a period")
def create_period(
    body: PeriodCreate,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> Period:
    existing = db.scalars(select(Period).where(Period.name == body.name)).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"period name {body.name!r} already exists")
    # validate start<end already done by schema, but double-check for partial updates
    period = Period(
        name=body.name.strip(),
        start_time=body.start_time,
        end_time=body.end_time,
        days_bitmask=body.days_bitmask,
        section=body.section,
    )
    db.add(period)
    db.commit()
    db.refresh(period)
    return period


@router.get("", response_model=list[PeriodOut], summary="List periods")
def list_periods(
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> list[Period]:
    return list(db.scalars(select(Period).order_by(Period.start_time.asc())).all())


@router.get("/{period_id}", response_model=PeriodOut, summary="Get one period")
def get_period(
    period_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> Period:
    return _get_or_404(db, period_id)


@router.put("/{period_id}", response_model=PeriodOut, summary="Update a period")
def update_period(
    period_id: int,
    body: PeriodUpdate,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> Period:
    period = _get_or_404(db, period_id)
    fields = body.model_dump(exclude_unset=True)
    if "name" in fields and fields["name"] != period.name:
        clash = db.scalars(select(Period).where(Period.name == fields["name"])).first()
        if clash is not None:
            raise HTTPException(status_code=409, detail=f"period name {fields['name']!r} already exists")
        fields["name"] = fields["name"].strip()
    # handle start/end cross-check when only one is provided
    start = fields.get("start_time", period.start_time)
    end = fields.get("end_time", period.end_time)
    if start >= end:
        raise HTTPException(status_code=422, detail="start_time must be before end_time")
    for k, v in fields.items():
        setattr(period, k, v)
    db.commit()
    db.refresh(period)
    return period


@router.delete("/{period_id}", status_code=status.HTTP_200_OK, summary="Delete a period")
def delete_period(
    period_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> dict:
    period = _get_or_404(db, period_id)
    db.delete(period)
    db.commit()
    return {"ok": True, "detail": f"period {period_id} deleted"}
