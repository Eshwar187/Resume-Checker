"""
Pydantic schemas for request/response models.
"""

from pydantic import BaseModel, EmailStr, validator, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles."""
    STUDENT = "student"
    PLACEMENT_TEAM = "placement_team"
    ADMIN = "admin"


class EvaluationVerdict(str, Enum):
    """Evaluation verdict."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# Base schemas
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: str = "Operation completed successfully"


class ErrorResponse(BaseModel):
    """Error response model."""
    error: bool = True
    message: str
    details: Optional[Dict[str, Any]] = None
    status_code: int


# Authentication schemas
class UserSignupRequest(BaseModel):
    """User signup request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=2, max_length=100)
    role: UserRole = UserRole.STUDENT
    
    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserLoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    name: str
    role: UserRole
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# File upload schemas
class FileUploadResponse(BaseModel):
    """File upload response."""
    file_id: str
    file_name: str
    file_url: str
    file_size: int
    upload_status: str = "success"


# Resume schemas
class ResumeUploadRequest(BaseModel):
    """Resume upload request metadata."""
    file_name: str
    file_size: int
    content_type: str
    
    @validator("file_size")
    def validate_file_size(cls, v):
        max_size = 10 * 1024 * 1024  # 10MB
        if v > max_size:
            raise ValueError("File size cannot exceed 10MB")
        return v
    
    @validator("content_type")
    def validate_content_type(cls, v):
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain"
        ]
        if v not in allowed_types:
            raise ValueError("File type not supported")
        return v


class ResumeResponse(BaseModel):
    """Resume response model."""
    id: str
    user_id: str
    file_name: str
    file_url: str
    file_size: int
    file_type: str
    extracted_skills: Optional[List[str]] = None
    uploaded_at: datetime
    
    class Config:
        from_attributes = True


# Job Description schemas
class JobDescriptionUploadRequest(BaseModel):
    """Job description upload request."""
    title: str = Field(..., min_length=1, max_length=200)
    company: Optional[str] = Field(None, max_length=100)
    job_text: Optional[str] = None  # For direct text input
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    content_type: Optional[str] = None


class JobDescriptionResponse(BaseModel):
    """Job description response."""
    id: str
    title: str
    company: Optional[str]
    job_title: Optional[str]
    must_have_skills: Optional[List[str]]
    good_to_have_skills: Optional[List[str]]
    qualifications: Optional[List[str]]
    experience_required: Optional[str]
    location: Optional[str]
    job_type: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Evaluation schemas
class EvaluationRequest(BaseModel):
    """Evaluation request."""
    resume_id: str
    jd_id: str
    use_advanced_ai: bool = True  # Whether to use LLM for evaluation


class EvaluationResponse(BaseModel):
    """Evaluation response."""
    id: str
    resume_id: str
    jd_id: str
    relevance_score: float = Field(..., ge=0, le=100)
    verdict: EvaluationVerdict
    matched_skills: List[str]
    missing_skills: List[str]
    skill_match_percentage: Optional[float]
    experience_match: bool
    experience_gap: Optional[str]
    relevant_projects: Optional[List[str]]
    missing_projects: Optional[List[str]]
    feedback: str
    recommendations: Optional[List[str]]
    processing_time: float
    created_at: datetime
    
    class Config:
        from_attributes = True


# Dashboard schemas
class StudentDashboardResponse(BaseModel):
    """Student dashboard response."""
    user: UserResponse
    resumes: List[ResumeResponse]
    evaluations: List[EvaluationResponse]
    total_resumes: int
    average_score: Optional[float]
    recent_activity: List[Dict[str, Any]]


class PlacementDashboardResponse(BaseModel):
    """Placement team dashboard response."""
    job_description: JobDescriptionResponse
    total_candidates: int
    evaluated_candidates: int
    average_score: Optional[float]
    score_distribution: Dict[str, int]  # High, Medium, Low counts
    top_candidates: List[Dict[str, Any]]
    recent_evaluations: List[EvaluationResponse]


class CandidateFilter(BaseModel):
    """Candidate filtering options."""
    min_score: Optional[float] = Field(None, ge=0, le=100)
    max_score: Optional[float] = Field(None, ge=0, le=100)
    verdict: Optional[List[EvaluationVerdict]] = None
    required_skills: Optional[List[str]] = None
    experience_match: Optional[bool] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


# Search and filter schemas
class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., min_length=1)
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class SearchResponse(BaseModel):
    """Search response."""
    results: List[Dict[str, Any]]
    total_count: int
    query: str
    execution_time: float


# Analytics schemas
class AnalyticsRequest(BaseModel):
    """Analytics request."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_id: Optional[str] = None
    jd_id: Optional[str] = None


class AnalyticsResponse(BaseModel):
    """Analytics response."""
    total_evaluations: int
    average_processing_time: float
    score_distribution: Dict[str, int]
    popular_skills: List[Dict[str, Any]]
    trends: Dict[str, Any]


# System schemas
class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    environment: str
    version: str
    timestamp: datetime
    services: Dict[str, str]


# Update forward references
TokenResponse.model_rebuild()