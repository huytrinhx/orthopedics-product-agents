"""Password hashing and JWT issuance/verification.

JWT_SECRET must be set in production (see .env.example); the local fallback
below only exists so a fresh clone doesn't hard-fail before `.env` is filled
in, matching this repo's existing convention of env vars with sensible local
defaults (e.g. NEO4J_PASSWORD).
"""
import os
import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "43200"))  # 30 days


def _jwt_secret() -> str:
    return os.environ.get("JWT_SECRET", "dev-only-insecure-secret-do-not-use-in-production")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(user_id: uuid.UUID) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> uuid.UUID:
    """Raises jwt.PyJWTError (expired, malformed, bad signature) on failure —
    callers (see auth/dependencies.py) turn that into a 401.
    """
    payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    return uuid.UUID(payload["sub"])


def create_oauth_state() -> str:
    """A short-lived, signed CSRF token for the Google OAuth redirect round
    trip (backend/auth/oauth.py) -- there's no server-side session to stash
    a nonce in between /google/login and /google/callback, so the state
    value itself has to prove it wasn't forged or replayed from stale.
    """
    now = datetime.now(UTC)
    payload = {"purpose": "oauth_state", "iat": now, "exp": now + timedelta(minutes=10)}
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def verify_oauth_state(state: str) -> bool:
    try:
        payload = jwt.decode(state, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return False
    return payload.get("purpose") == "oauth_state"


def is_allowlisted_admin(email: str) -> bool:
    allowlist = {
        e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()
    }
    return email.strip().lower() in allowlist
