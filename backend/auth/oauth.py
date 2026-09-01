"""Google OAuth (authorization-code flow). One extra way to get a session
alongside password login (auth/security.py's JWT) -- the rest of the app
never knows which path a user came in through.
"""
import os
from urllib.parse import urlencode

import httpx

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def _redirect_uri() -> str:
    # Must exactly match an authorized redirect URI on the Google Cloud
    # OAuth client (GOOGLE_CLIENT_ID) -- see README's Google OAuth setup note.
    base = os.environ.get("OAUTH_REDIRECT_BASE_URL", "http://localhost:8000")
    return f"{base}/auth/google/callback"


def build_authorize_url(state: str) -> str:
    params = {
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": "openid email",
        "state": state,
        "prompt": "select_account",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_access_token(code: str) -> str:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            TOKEN_URL,
            data={
                "client_id": os.environ["GOOGLE_CLIENT_ID"],
                "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                "code": code,
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
            },
        )
        res.raise_for_status()
        return res.json()["access_token"]


async def fetch_verified_email(access_token: str) -> str:
    async with httpx.AsyncClient() as client:
        res = await client.get(
            USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}
        )
        res.raise_for_status()
        info = res.json()
    if not info.get("email_verified"):
        raise ValueError("Google account email is not verified")
    return info["email"]
