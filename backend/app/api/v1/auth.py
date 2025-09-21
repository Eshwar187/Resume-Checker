from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
import logging

from app.schemas import (
    UserSignupRequest, UserLoginRequest, TokenResponse, 
    BaseResponse, UserResponse
)
from app.core.auth import get_auth_manager, get_current_user, security

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: UserSignupRequest):
    """
    Register a new user account.
    
    - **email**: User's email address
    - **password**: User's password (minimum 8 characters)
    - **name**: User's full name
    - **role**: User role (student or placement_team)
    """
    try:
        auth_manager = get_auth_manager()
        result = await auth_manager.sign_up(
            email=request.email,
            password=request.password,
            name=request.name,
            role=request.role
        )
        
        logger.info(f"New user registered: {request.email}")
        return TokenResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest):
    """
    Authenticate user and return access token.
    
    - **email**: User's email address
    - **password**: User's password
    """
    try:
        auth_manager = get_auth_manager()
        result = await auth_manager.sign_in(
            email=request.email,
            password=request.password
        )
        
        logger.info(f"User logged in: {request.email}")
        return TokenResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


@router.post("/logout", response_model=BaseResponse)
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout user and invalidate token.
    """
    try:
        auth_manager = get_auth_manager()
        success = await auth_manager.sign_out(credentials.credentials)
        
        if success:
            return BaseResponse(message="Logged out successfully")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logout failed"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    return current_user


@router.get("/verify", response_model=BaseResponse)
async def verify_token(current_user: UserResponse = Depends(get_current_user)):
    """
    Verify if the current token is valid.
    """
    return BaseResponse(
        message=f"Token is valid for user: {current_user.email}"
    )