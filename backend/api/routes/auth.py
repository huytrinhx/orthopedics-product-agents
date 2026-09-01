"""Signup/login/current-user, plus Google OAuth as a second way in. is_admin
is decided once, at (first) signup, by checking the account's email against
the ADMIN_EMAILS allowlist — see agents.md's "Standing technical decisions".
"""
import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

from auth import oauth
from auth.dependencies import get_current_user
from auth.models import LoginRequest, SignupRequest, TokenResponse, UserOut
from auth.repository import UserRecord, create_user, get_user_by_email
from auth.security import (
    create_access_token,
    create_oauth_state,
    hash_password,
    is_allowlisted_admin,
    verify_oauth_state,
    verify_password,
)

router = APIRouter()


def _user_out(user: UserRecord) -> UserOut:
    return UserOut(
        id=user.id, email=user.email, is_admin=user.is_admin, created_at=user.created_at
    )


@router.post("/signup", response_model=TokenResponse)
async def signup(body: SignupRequest) -> TokenResponse:
    if await get_user_by_email(body.email) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")

    user = await create_user(
        email=body.email,
        hashed_password=hash_password(body.password),
        is_admin=is_allowlisted_admin(body.email),
    )
    return TokenResponse(access_token=create_access_token(user.id), user=_user_out(user))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    user = await get_user_by_email(body.email)
    if user is None or user.hashed_password is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")

    return TokenResponse(access_token=create_access_token(user.id), user=_user_out(user))


@router.get("/me", response_model=UserOut)
async def me(user: UserRecord = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


@router.get("/google/login")
async def google_login() -> RedirectResponse:
    return RedirectResponse(oauth.build_authorize_url(create_oauth_state()))


@router.get("/google/callback")
async def google_callback(code: str, state: str) -> RedirectResponse:
    if not verify_oauth_state(state):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired OAuth state")

    access_token = await oauth.exchange_code_for_access_token(code)
    email = await oauth.fetch_verified_email(access_token)

    user = await get_user_by_email(email)
    if user is None:
        # hashed_password=None: an OAuth-only account, matches the users
        # table's nullable column (see migrations/versions/..._create_users_table.py).
        # Signing in with an email that already has a password account
        # (the `if user is None` above being False) links to that same row
        # instead of creating a duplicate.
        user = await create_user(email=email, hashed_password=None, is_admin=is_allowlisted_admin(email))

    session_token = create_access_token(user.id)
    frontend_base = os.environ.get("FRONTEND_PUBLIC_URL", "")
    return RedirectResponse(f"{frontend_base}/?auth_token={session_token}")
