"""FastAPI dependencies for authenticated/admin-gated routes.

get_current_user is the one every authenticated route (chat, feedback, ...)
depends on; require_admin builds on it for the documents/evals/admin routes
that ticket 04+ add.
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
