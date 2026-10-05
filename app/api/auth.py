"""Auth endpoints.

Public
------
``POST /auth/token``           username/password -> JWT
``POST /auth/register``        self sign-up -> *pending* account
``POST /auth/forgot-password`` raise a reset request the super admin resolves
``GET  /auth/me``              the signed-in principal

Super admin only (``/auth/admin/*``)
------------------------------------
Account review (approve / reject / suspend), credential changes (the only
place a username or password may be set) and forgotten-password resolution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..models import PasswordResetRequest, User
from ..schemas import (
    ForgotPasswordOut,
    ForgotPasswordRequest,
    MeOut,
    OkResponse,
    PasswordRequestListResponse,
    PasswordRequestOut,
    PasswordResolveRequest,
    RegisterRequest,
    RegisteredOut,
    TokenRequest,
    TokenResponse,
    UserApprove,
    UserListResponse,
    UserOut,
    UserUpdate,
)
from ..security import (
    Principal,
    create_access_token,
    get_current_user,
    require_super_admin,
)
from ..users import (
    ROLE_SUPER,
    STATUS_ACTIVE,
    authenticate,
    count_active_super_admins,
    ensure_seed_superuser,
    get_reset_request,
    get_user,
    get_user_by_username,
    hash_password,
    list_reset_requests,
    list_users,
    password_ok,
    user_status_error,
    username_ok,
)
from .deps import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

DbDep = Annotated[Session, Depends(get_db)]
SuperDep = Annotated[Principal, Depends(require_super_admin)]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _user_out(user) -> UserOut:
    return UserOut.model_validate(user)


# ------------------------------------------------------------------- public
@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Exchange credentials for a JWT",
)
def issue_token(body: TokenRequest, db: DbDep) -> TokenResponse:
    ensure_seed_superuser(db)
    user = authenticate(db, body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    reason = user_status_error(user)
    if reason is not None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=reason)
    token, expires_in = create_access_token(
        user.username, role=user.role, user_id=user.id
    )
    return TokenResponse(
        access_token=token, expires_in=expires_in, role=user.role
    )


@router.post(
    "/register",
    response_model=RegisteredOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register an account (pending super-admin approval)",
)
def register(body: RegisterRequest, db: DbDep) -> RegisteredOut:
    ensure_seed_superuser(db)
    if not username_ok(body.username):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="username must be 3-100 chars of letters, digits, '_', '.', '-'",
        )
    if not password_ok(body.password):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="password must be at least 8 characters",
        )
    if get_user_by_username(db, body.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="username already taken",
        )

    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        display_name=body.display_name or body.username,
        role=body.requested_role,
        status="pending",
        security_level=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return RegisteredOut(
        id=user.id,
        username=user.username,
        status=user.status,
        requested_role=body.requested_role,
        detail=(
            "Account created and queued for review. The super admin must "
            "approve it before you can sign in."
        ),
    )


@router.post(
    "/forgot-password",
    response_model=ForgotPasswordOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ask the super admin to reset a password",
)
def forgot_password(body: ForgotPasswordRequest, db: DbDep) -> ForgotPasswordOut:
    ensure_seed_superuser(db)
    # Always accepted with the same wording so this endpoint never confirms
    # whether an account exists.
    db.add(
        PasswordResetRequest(
            username=body.username,
            reason=body.reason,
            status="pending",
        )
    )
    db.commit()
    return ForgotPasswordOut(
        accepted=True,
        detail=(
            "Request recorded. A super admin will review it and set a new "
            "password; you will be able to sign in once they do."
        ),
    )


@router.get(
    "/me",
    response_model=MeOut,
    summary="The signed-in principal (username, role, security level)",
)
def me(
    principal: Annotated[Principal, Depends(get_current_user)], db: DbDep
) -> MeOut:
    row = get_user(db, principal.user_id) if principal.user_id else None
    if row is None:
        row = get_user_by_username(db, principal.username)
    if row is None:
        return MeOut(
            username=principal.username,
            role=principal.role,
            status=STATUS_ACTIVE,
            security_level=principal.security_level,
        )
    return MeOut(
        id=row.id,
        username=row.username,
        role=row.role,
        status=row.status,
        display_name=row.display_name,
        security_level=row.security_level,
    )


# ------------------------------------------------- super admin: credentials
@router.get(
    "/admin/users",
    response_model=UserListResponse,
    summary="List every account (super admin only)",
)
def admin_list_users(super: SuperDep, db: DbDep) -> UserListResponse:
    rows = list_users(db)
    return UserListResponse(items=[_user_out(u) for u in rows], total=len(rows))


@router.patch(
    "/admin/users/{user_id}",
    response_model=UserOut,
    summary="Set username / password / role / level (super admin only)",
)
def admin_update_user(
    user_id: int, body: UserUpdate, super: SuperDep, db: DbDep
) -> UserOut:
    target = get_user(db, user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"user {user_id} not found",
        )

    # Never let an edit remove the last usable super admin.
    loses_super = target.role == ROLE_SUPER and (
        (body.role is not None and body.role != ROLE_SUPER)
        or (body.status is not None and body.status != STATUS_ACTIVE)
    )
    if loses_super and count_active_super_admins(db, exclude_id=target.id) == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="cannot demote or disable the last active super admin",
        )

    if body.username is not None and body.username != target.username:
        other = get_user_by_username(db, body.username)
        if other is not None and other.id != target.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="username already taken",
            )
        target.username = body.username
    if body.password is not None:
        target.password_hash = hash_password(body.password)
        target.password_source = "manual"
    if body.role is not None:
        target.role = body.role
    if body.security_level is not None:
        target.security_level = body.security_level
    if body.status is not None:
        target.status = body.status

    db.commit()
    db.refresh(target)
    return _user_out(target)


@router.post(
    "/admin/users/{user_id}/approve",
    response_model=UserOut,
    summary="Approve a registered account (super admin only)",
)
def admin_approve_user(
    user_id: int, body: UserApprove, super: SuperDep, db: DbDep
) -> UserOut:
    target = get_user(db, user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"user {user_id} not found"
        )
    target.status = STATUS_ACTIVE
    target.security_level = body.security_level
    target.approved_at = _now()
    target.approved_by = super.username
    db.commit()
    db.refresh(target)
    return _user_out(target)


def _set_status(super: SuperDep, db: DbDep, user_id: int, new_status: str) -> UserOut:
    target = get_user(db, user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"user {user_id} not found"
        )
    if (
        target.role == ROLE_SUPER
        and new_status != STATUS_ACTIVE
        and count_active_super_admins(db, exclude_id=target.id) == 0
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="cannot disable the last active super admin",
        )
    target.status = new_status
    db.commit()
    db.refresh(target)
    return _user_out(target)


@router.post(
    "/admin/users/{user_id}/reject",
    response_model=UserOut,
    summary="Reject a pending account (super admin only)",
)
def admin_reject_user(user_id: int, super: SuperDep, db: DbDep) -> UserOut:
    return _set_status(super, db, user_id, "rejected")


@router.post(
    "/admin/users/{user_id}/suspend",
    response_model=UserOut,
    summary="Suspend an approved account (super admin only)",
)
def admin_suspend_user(user_id: int, super: SuperDep, db: DbDep) -> UserOut:
    return _set_status(super, db, user_id, "suspended")


# ----------------------------------------- super admin: password requests
@router.get(
    "/admin/password-requests",
    response_model=PasswordRequestListResponse,
    summary="Forgotten-password requests (super admin only)",
)
def admin_list_password_requests(
    super: SuperDep,
    db: DbDep,
    request_status: Annotated[str | None, Query(alias="status")] = None,
) -> PasswordRequestListResponse:
    rows = list_reset_requests(db, status=request_status)
    return PasswordRequestListResponse(
        items=[PasswordRequestOut.model_validate(r) for r in rows], total=len(rows)
    )


@router.post(
    "/admin/password-requests/{request_id}",
    response_model=OkResponse,
    summary="Resolve a request by setting the new password (super admin only)",
)
def admin_resolve_password_request(
    request_id: int, body: PasswordResolveRequest, super: SuperDep, db: DbDep
) -> OkResponse:
    req = get_reset_request(db, request_id)
    if req is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"password request {request_id} not found",
        )
    if req.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"request already {req.status}",
        )

    if body.action == "reject":
        req.status = "rejected"
        req.resolved_at = _now()
        req.resolved_by = super.username
        db.commit()
        return OkResponse(ok=True, detail="request rejected")

    target = get_user_by_username(db, req.username)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no account named '{req.username}' to reset",
        )
    target.password_hash = hash_password(body.new_password or "")
    target.password_source = "manual"
    req.status = "resolved"
    req.resolved_at = _now()
    req.resolved_by = super.username
    db.commit()
    return OkResponse(
        ok=True,
        detail=f"password reset for '{target.username}' by {super.username}",
    )
