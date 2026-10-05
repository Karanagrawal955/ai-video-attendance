"""JWT authentication, principals and role guards for admin/staff endpoints."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .users import (
    ROLE_SUPER,
    STATUS_ACTIVE,
    get_user,
    get_user_by_username,
    user_status_error,
)

_bearer = HTTPBearer(auto_error=False)


class Principal(NamedTuple):
    """The authenticated caller: username + role resolved from the DB."""

    username: str
    role: str
    user_id: int | None = None
    security_level: int = 0

    @property
    def is_super_admin(self) -> bool:
        return self.role == ROLE_SUPER


def create_access_token(
    subject: str, role: str = "admin", user_id: int | None = None
) -> tuple[str, int]:
    """Return (token, expires_in_seconds)."""
    now = datetime.now(timezone.utc)
    expires_in = settings.jwt_expire_minutes * 60
    payload: dict = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
        "typ": "access",
    }
    if user_id is not None:
        payload["uid"] = user_id
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:  # pragma: no cover - trivial
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def verify_credentials(username: str, password: str) -> bool:
    """Constant-time comparison against the .env admin credentials.

    Legacy helper kept for the seeded super admin; account logins go through
    ``app.users.authenticate`` (see the .env fallback there).
    """
    return secrets.compare_digest(
        username.encode("utf-8"), settings.admin_username.encode("utf-8")
    ) and secrets.compare_digest(
        password.encode("utf-8"), settings.admin_password.encode("utf-8")
    )


def resolve_principal(
    credentials: HTTPAuthorizationCredentials | None, db: Session
) -> Principal:
    """Turn a bearer token into a DB-backed :class:`Principal`."""
    if not settings.auth_required:
        # Auth disabled: every caller is treated as the super admin so the
        # console stays fully usable in local/prototype setups.
        return Principal(settings.admin_username, ROLE_SUPER, None, 5)
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    user_id = payload.get("uid")
    subject = str(payload.get("sub") or "")

    row = get_user(db, int(user_id)) if isinstance(user_id, int) else None
    if row is None and subject:
        row = get_user_by_username(db, subject)

    if row is None:
        # Token minted before the users table existed (or auth disabled earlier).
        role = ROLE_SUPER if subject == settings.admin_username else "admin"
        return Principal(subject or settings.admin_username, role, None, 5 if role == ROLE_SUPER else 0)

    reason = user_status_error(row)
    if reason is not None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=reason)
    return Principal(row.username, row.role, row.id, row.security_level)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Principal:
    """FastAPI dependency: authenticated principal (unless AUTH_REQUIRED=false)."""
    return resolve_principal(credentials, db)


def get_current_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> str:
    """FastAPI dependency: require a valid bearer token, return its username."""
    return resolve_principal(credentials, db).username


def require_super_admin(
    principal: Principal = Depends(get_current_user),
) -> Principal:
    """FastAPI dependency: super-admin-only endpoints (credential management)."""
    if not principal.is_super_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="super admin only",
        )
    return principal
