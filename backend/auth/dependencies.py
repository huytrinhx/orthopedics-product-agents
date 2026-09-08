"""FastAPI dependencies for authenticated/admin-gated routes.

get_current_user is the one every authenticated route (chat, feedback, ...)
depends on; require_admin builds on it for the documents/evals/admin routes
that ticket 04+ add. require_chat_access builds on it too, for the one
place a disabled (is_active=False) user is actually blocked -- see its
docstring below.
"""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.repository import UserRecord, get_user_by_id
from auth.security import decode_access_token

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserRecord:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        user_id = decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user


async def require_admin(user: UserRecord = Depends(get_current_user)) -> UserRecord:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user


async def require_chat_access(user: UserRecord = Depends(get_current_user)) -> UserRecord:
    """A brand-new signup starts is_active=False -- an admin has to enable
    them (see api/routes/users.py) before they can chat. Admins bypass this
    entirely (their is_active value is never even consulted): they're
    trusted by construction (ADMIN_EMAILS) and the Users tab shows their row
    as read-only, so there's nothing for this check to enforce against them.

    Deliberately narrower than require_admin's is_admin gate: login, /auth/me,
    and everything else stay open to a pending user so the frontend can show
    them an honest "pending approval" state instead of a dead end. This is
    also a live per-request check (get_current_user always re-reads the DB
    row, no caching), so disabling someone takes effect on their very next
    chat call, not just their next login.
    """
    if not user.is_admin and not user.is_active:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Your account is pending admin approval"
        )
    return user
