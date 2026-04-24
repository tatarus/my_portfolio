from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    password2: str
    first_name: str
    last_name: str
    patronymic: Optional[str] = None

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    patronymic: Optional[str] = None

class UserOut(BaseModel):
    id: int
    email: EmailStr
    first_name: str
    last_name: str
    patronymic: Optional[str] = None
    is_active: bool