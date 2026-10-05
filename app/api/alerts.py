"""Alert endpoints: list + acknowledge."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Alert
from ..schemas import AlertOut
from ..timeutil import utcnow
from .deps import get_current_admin, get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertOut], summary="List alerts (newest first)")
def list_alerts(
    type: str | None = Query(default=None, description="Filter by alert type"),
    severity: str | None = Query(default=None),
    acknowledged: bool | None = Query(default=None, description="true=only acked, false=only unacked"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> list[Alert]:
    stmt = select(Alert).order_by(Alert.created_at.desc()).offset(offset).limit(limit)
    if type is not None:
        stmt = stmt.where(Alert.type == type)
    if severity is not None:
        stmt = stmt.where(Alert.severity == severity)
    if acknowledged is True:
        stmt = stmt.where(Alert.acknowledged_at.isnot(None))
    elif acknowledged is False:
        stmt = stmt.where(Alert.acknowledged_at.is_(None))
    return list(db.scalars(stmt).all())


@router.get("/{alert_id}", response_model=AlertOut, summary="Get one alert")
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> Alert:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
    return alert


@router.post("/{alert_id}/ack", response_model=AlertOut, summary="Acknowledge an alert")
def ack_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> Alert:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
    if alert.acknowledged_at is None:
        alert.acknowledged_at = utcnow()
        db.commit()
        db.refresh(alert)
    return alert


@router.post("/{alert_id}/resolve", response_model=AlertOut, summary="Resolve an alert")
def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_admin),
) -> Alert:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
    if alert.resolved_at is None:
        alert.resolved_at = utcnow()
        if alert.acknowledged_at is None:
            alert.acknowledged_at = alert.resolved_at
        db.commit()
        db.refresh(alert)
    return alert
