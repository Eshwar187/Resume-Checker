from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from app.config import settings
from app.core.supabase import get_supabase_client, get_supabase_service_client
from app.schemas import UserResponse, UserRole

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Security
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.jwt_secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(
            token, 
            settings.jwt_secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_access_token(token: str) -> Dict[str, Any]:
    """Alias for verify_token - verify and decode JWT access token."""
    return verify_token(token)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """Get current authenticated user."""
    try:
        # Verify JWT token
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
        # Get user from Supabase
        supabase = get_supabase_service_client()
        result = supabase.table("users").select("*").eq("id", user_id).single().execute()
        
        if result.data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        user_data = result.data
        return UserResponse(**user_data)
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_role: UserRole):
    """Decorator to require specific user role."""
    def role_checker(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires {required_role.value} role"
            )
        return current_user
    
    return role_checker


def require_roles(required_roles: list[UserRole]):
    """Decorator to require one of multiple roles."""
    def role_checker(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        if current_user.role not in required_roles and current_user.role != UserRole.ADMIN:
            role_names = [role.value for role in required_roles]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of these roles: {', '.join(role_names)}"
            )
        return current_user
    
    return role_checker


class SupabaseAuth:
    """Supabase authentication manager."""
    
    def __init__(self):
        self.client = get_supabase_client()
        self.service_client = get_supabase_service_client()
    
    async def sign_up(self, email: str, password: str, name: str, role: UserRole = UserRole.STUDENT) -> Dict[str, Any]:
        """Sign up a new user."""
        try:
            # Create user in Supabase Auth
            auth_result = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "name": name,
                        "role": role.value
                    }
                }
            })
            
            if auth_result.user is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to create user account"
                )
            
            user_id = auth_result.user.id
            
            # Create user record in database
            user_data = {
                "id": user_id,
                "email": email,
                "name": name,
                "role": role.value,
                "is_active": True
            }
            
            db_result = self.service_client.table("users").insert(user_data).execute()
            
            if db_result.data is None:
                # Rollback auth user creation if database insert fails
                logger.error("Failed to create user record in database")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to complete user registration"
                )
            
            # Create JWT token
            access_token = create_access_token(data={"sub": user_id, "email": email, "role": role.value})
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.access_token_expire_minutes * 60,
                "user": UserResponse(**db_result.data[0])
            }
            
        except Exception as e:
            logger.error(f"Sign up error: {e}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in a user."""
        try:
            # Authenticate with Supabase
            auth_result = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if auth_result.user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            user_id = auth_result.user.id
            
            # Get user data from database
            db_result = self.service_client.table("users").select("*").eq("id", user_id).single().execute()
            
            if db_result.data is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User profile not found"
                )
            
            user_data = db_result.data
            
            # Check if user is active
            if not user_data.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated"
                )
            
            # Create JWT token
            access_token = create_access_token(
                data={
                    "sub": user_id, 
                    "email": email, 
                    "role": user_data["role"]
                }
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.access_token_expire_minutes * 60,
                "user": UserResponse(**user_data)
            }
            
        except Exception as e:
            logger.error(f"Sign in error: {e}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    async def sign_out(self, token: str) -> bool:
        """Sign out a user."""
        try:
            # In a production app, you might want to blacklist the token
            # For now, we'll just validate it exists
            verify_token(token)
            return True
        except Exception as e:
            logger.error(f"Sign out error: {e}")
            return False


# Global auth instance
auth_manager = SupabaseAuth()


def get_auth_manager() -> SupabaseAuth:
    """Get the authentication manager."""
    return auth_manager