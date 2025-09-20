"""
Core configuration module for the Resume Relevance Check System.
Handles environment variables and application settings.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Application
    app_name: str = "Resume Relevance Check API"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Supabase
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    
    # Database
    database_url: Optional[str] = None
    
    # JWT
    jwt_secret_key: Optional[str] = "default-secret-key-for-testing-only-change-in-production"
    secret_key: Optional[str] = "default-secret-key-for-testing-only-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    # AI/LLM APIs
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "resume-relevance-checker"
    
    # Vector Database
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    
    # File Upload
    max_file_size: int = 10485760  # 10MB
    
    # Redis (optional)
    redis_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    @property
    def allowed_extensions(self) -> List[str]:
        """Get allowed file extensions."""
        return ["pdf", "docx", "doc", "txt"]
    
    @property
    def allowed_origins(self) -> List[str]:
        """Get allowed origins."""
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ]
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins."""
        return self.allowed_origins

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings