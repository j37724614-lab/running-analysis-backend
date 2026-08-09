import datetime
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import bcrypt
import jwt

from db.session import get_session
from db_models import User
from sqlmodel import SQLModel
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET_KEY = "running_analysis_secret_key_change_me_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

from typing import Optional
from fastapi import Query

security_bearer = HTTPBearer(auto_error=False)

class RegisterIn(BaseModel):
    username: str
    password: str

class LoginIn(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    token: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    resolved_token = None
    if credentials and credentials.credentials:
        resolved_token = credentials.credentials
    elif token:
        resolved_token = token
        
    if not resolved_token:
        raise credentials_exception
        
    try:
        payload = jwt.decode(resolved_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = (await session.execute(
        select(User).where(User.username == username)
    )).scalars().first()
    
    if user is None:
        raise credentials_exception
    return user

@router.post("/register", response_model=dict)
async def register(
    data: RegisterIn,
    session: AsyncSession = Depends(get_session)
):
    # Check if username exists
    existing_user = (await session.execute(
        select(User).where(User.username == data.username)
    )).scalars().first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
        
    hashed = hash_password(data.password)
    new_user = User(username=data.username, hashed_password=hashed)
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    
    return {"message": "User registered successfully", "id": str(new_user.id)}

@router.post("/login", response_model=TokenOut)
async def login(
    data: LoginIn,
    session: AsyncSession = Depends(get_session)
):
    user = (await session.execute(
        select(User).where(User.username == data.username)
    )).scalars().first()
    
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return TokenOut(access_token=access_token, token_type="bearer")
