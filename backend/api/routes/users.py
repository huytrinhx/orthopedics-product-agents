"""Admin-only user management, backing the Users tab: list every account and
flip a non-admin's is_active flag (see auth.dependencies.require_chat_access
for where that flag is actually enforced). Admin rows are read-only here --
is_admin is decided once at signup from ADMIN_EMAILS (agents.md's "Standing
technical decisions") and this API never changes it.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import require_admin
from auth.models import SetUserActiveRequest, UserOut
from auth.repository import UserRecord, get_user_by_id, list_users, set_user_active

router = APIRouter()


def _user_out(user: UserRecord) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        is_admin=user.is_admin,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/", response_model=list[UserOut])
async def list_all_users(admin: UserRecord = Depends(require_admin)) -> list[UserOut]:
    return [_user_out(u) for u in await list_users()]


@router.patch("/{user_id}/active", response_model=UserOut)
async def set_active(
    user_id: uuid.UUID,
    body: SetUserActiveRequest,
    admin: UserRecord = Depends(require_admin),
) -> UserOut:
    target = await get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    # Defense in depth: the Users tab already hides this action for admin
    # rows, but require_chat_access never even looks at is_active for an
    # admin, so honoring a request to change it here would silently do
    # nothing useful while implying it did.
    if target.is_admin:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Admin accounts can't be disabled")
    updated = await set_user_active(user_id, body.is_active)
    return _user_out(updated)
