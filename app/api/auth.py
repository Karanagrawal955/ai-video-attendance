"""Auth endpoints: POST /auth/token (JSON username/password -> JWT)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import TokenRequest, TokenResponse
from ..security import create_access_token, verify_credentials

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Exchange admin credentials for a JWT",
)
def issue_token(body: TokenRequest) -> TokenResponse:
    if not verify_credentials(body.username, body.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(body.username)
    return TokenResponse(access_token=token, expires_in=expires_in)
