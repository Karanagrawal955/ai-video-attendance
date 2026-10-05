"""Multi-user accounts: roles, approval status, password hashing, seeding.

Roles
-----
``super_admin``
    The ONLY role allowed to set or change a username/password, approve or
    reject registrations, and resolve forgotten-password requests.
``admin``
    Regular operator.  Reads its own profile only; it can never edit
    credentials (its own included).

Statuses
--------
``pending``    registered, waiting for super-admin review (cannot log in)
``active``     approved by the super admin (may log in)
``rejected``   refused by the super admin
``suspended``  approved once, later disabled by the super admin

Bootstrap
---------
``ensure_seed_superuser`` materialises the very first super admin from
``ADMIN_USERNAME`` / ``ADMIN_PASSWORD`` on first use, so a fresh install keeps
working with the documented ``admin`` / ``admin`` login.  It runs *before* any
account can be created, which is why a self-registered account can never claim
the seeded username.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .models import PasswordResetRequest, User

ROLE_SUPER = "super_admin"
ROLE_ADMIN = "admin"
ROLES: tuple[str, ...] = (ROLE_SUPER, ROLE_ADMIN)

STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"
STATUS_SUSPENDED = "suspended"
STATUSES: tuple[str, ...] = (
    STATUS_PENDING,
    STATUS_ACTIVE,
    STATUS_REJECTED,
    STATUS_SUSPENDED,
)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,100}$")
PASSWORD_MIN_LENGTH = 8

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 100_000
# Upper bound so a corrupted row can never turn a login into a CPU bomb.
_MAX_ITERATIONS = 5_000_000


# ----------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    """PBKDF2-SHA256 with a per-password random salt.

    Encoded as ``pbkdf2_sha256$<iterations>$<salt>$<digest>`` (urlsafe base64)
    so parameters stay attached to the hash and can be upgraded later.
    """
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _ITERATIONS
    )
    return "$".join(
        (
            _ALGO,
            str(_ITERATIONS),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time password check; malformed hashes simply fail closed."""
    try:
        algo, iterations, salt_b64, digest_b64 = str(encoded).split("$", 3)
        if algo != _ALGO:
            return False
        iterations_i = int(iterations)
        if not 1 <= iterations_i <= _MAX_ITERATIONS:
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
    except (ValueError, TypeError, UnicodeError):
        return False
    if not password or len(password) > 1024:
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations_i
    )
    return hmac.compare_digest(actual, expected)


def username_ok(username: str) -> bool:
    return bool(USERNAME_RE.match(username or ""))


def password_ok(password: str) -> bool:
    return bool(password) and len(password) <= 1024 and len(password) >= PASSWORD_MIN_LENGTH


# ------------------------------------------------------------------ queries
def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))


def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def list_users(db: Session) -> list[User]:
    # super admins first, then newest accounts
    return list(
        db.scalars(
            select(User).order_by(
                (User.role != ROLE_SUPER).desc(), User.created_at.desc()
            )
        )
    )


def count_active_super_admins(db: Session, exclude_id: int | None = None) -> int:
    stmt = select(User).where(
        User.role == ROLE_SUPER, User.status == STATUS_ACTIVE
    )
    if exclude_id is not None:
        stmt = stmt.where(User.id != exclude_id)
    return len(list(db.scalars(stmt)))


def list_reset_requests(
    db: Session, status: str | None = None
) -> list[PasswordResetRequest]:
    stmt = select(PasswordResetRequest).order_by(
        PasswordResetRequest.created_at.desc()
    )
    if status:
        stmt = stmt.where(PasswordResetRequest.status == status)
    return list(db.scalars(stmt))


def get_reset_request(db: Session, request_id: int) -> PasswordResetRequest | None:
    return db.get(PasswordResetRequest, request_id)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ bootstrap
def ensure_seed_superuser(db: Session) -> User | None:
    """Create the first super admin from ``ADMIN_USERNAME``/``ADMIN_PASSWORD``.

    Idempotent: any existing super admin (even a renamed one) wins, and a
    username already taken by somebody else is never silently promoted.
    """
    existing = db.scalar(select(User).where(User.role == ROLE_SUPER).limit(1))
    if existing is not None:
        return existing

    username = settings.admin_username
    if not username_ok(username):
        username = "admin"
    if get_user_by_username(db, username) is not None:
        db.rollback()
        return None  # taken by a pending/other account: never promote it

    user = User(
        username=username,
        password_hash=hash_password(settings.admin_password),
        password_source="env",
        display_name="Super Admin",
        role=ROLE_SUPER,
        status=STATUS_ACTIVE,
        security_level=5,
        approved_at=_now(),
        approved_by="bootstrap",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# -------------------------------------------------------------------- login
def authenticate(db: Session, username: str, password: str) -> User | None:
    """Return the matching **active** user, else ``None``.

    A pending/rejected/suspended account still authenticates here so the caller
    can answer with a precise 403 instead of a misleading 401.
    """
    user = get_user_by_username(db, username)
    if user is None:
        return None
    if verify_password(password, user.password_hash):
        return user

    # .env fallback: while the password still mirrors ADMIN_PASSWORD, rotating
    # that variable keeps unlocking the seeded super admin (and re-syncs the
    # hash).  Once the super admin sets a password through the API, the source
    # flips to 'manual' and .env can never override it again.
    if (
        user.role == ROLE_SUPER
        and user.status == STATUS_ACTIVE
        and user.password_source == "env"
        and hmac.compare_digest(username.encode("utf-8"), settings.admin_username.encode("utf-8"))
        and hmac.compare_digest(password.encode("utf-8"), settings.admin_password.encode("utf-8"))
    ):
        user.password_hash = hash_password(password)
        db.commit()
        return user
    return None


def user_status_error(user: User) -> str | None:
    """Human-readable reason this account may not sign in, or ``None``."""
    if user.status == STATUS_ACTIVE:
        return None
    if user.status == STATUS_PENDING:
        return "account awaiting approval by the super admin"
    if user.status == STATUS_REJECTED:
        return "account registration was rejected by the super admin"
    return "account suspended — ask the super admin"
