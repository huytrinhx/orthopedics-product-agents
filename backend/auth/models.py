"""Pydantic request/response shapes for the auth routes."""
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    is_admin: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    user: UserOut
