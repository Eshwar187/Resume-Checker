"""
Database models for the Resume Relevance Check System.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import List, Optional
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """User roles enum."""
    STUDENT = "student"
    PLACEMENT_TEAM = "placement_team"
    ADMIN = "admin"


class EvaluationVerdict(str, enum.Enum):
    """Evaluation verdict enum."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)  # Supabase UUID
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False, default=UserRole.STUDENT.value)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    resumes = relationship("Resume", back_populates="user")
    job_descriptions = relationship("JobDescription", back_populates="uploader")


class Resume(Base):
    """Resume model."""
    __tablename__ = "resumes"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    file_name = Column(String, nullable=False)
    file_url = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)
    raw_text = Column(Text)
    processed_text = Column(Text)
    extracted_skills = Column(JSON)  # List of extracted skills
    extracted_experience = Column(JSON)  # Experience details
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="resumes")
    evaluations = relationship("Evaluation", back_populates="resume")


class JobDescription(Base):
    """Job Description model."""
    __tablename__ = "job_descriptions"
    
    id = Column(String, primary_key=True)
    uploader_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    company = Column(String)
    file_name = Column(String)
    file_url = Column(String)
    raw_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    
    # Extracted information
    job_title = Column(String)
    must_have_skills = Column(JSON)  # List of required skills
    good_to_have_skills = Column(JSON)  # List of preferred skills
    qualifications = Column(JSON)  # List of qualifications
    experience_required = Column(String)
    location = Column(String)
    job_type = Column(String)  # Full-time, Part-time, Contract, etc.
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    uploader = relationship("User", back_populates="job_descriptions")
    evaluations = relationship("Evaluation", back_populates="job_description")


class Evaluation(Base):
    """Evaluation results model."""
    __tablename__ = "evaluations"
    
    id = Column(String, primary_key=True)
    resume_id = Column(String, ForeignKey("resumes.id"), nullable=False)
    jd_id = Column(String, ForeignKey("job_descriptions.id"), nullable=False)
    
    # Evaluation results
    relevance_score = Column(Float, nullable=False)  # 0-100
    verdict = Column(String, nullable=False)  # High, Medium, Low
    
    # Skill analysis
    matched_skills = Column(JSON)  # List of matched skills
    missing_skills = Column(JSON)  # List of missing skills
    skill_match_percentage = Column(Float)  # Percentage of skills matched
    
    # Experience analysis
    experience_match = Column(Boolean, default=False)
    experience_gap = Column(String)  # Description of experience gap
    
    # Project analysis
    relevant_projects = Column(JSON)  # List of relevant projects found
    missing_projects = Column(JSON)  # Suggested project types
    
    # Feedback and recommendations
    feedback = Column(Text)  # Detailed feedback
    recommendations = Column(JSON)  # List of improvement recommendations
    
    # Technical details
    hard_match_score = Column(Float)  # Keyword-based score
    soft_match_score = Column(Float)  # Semantic similarity score
    processing_time = Column(Float)  # Time taken for evaluation in seconds
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    resume = relationship("Resume", back_populates="evaluations")
    job_description = relationship("JobDescription", back_populates="evaluations")


class EvaluationLog(Base):
    """Evaluation processing logs for debugging and monitoring."""
    __tablename__ = "evaluation_logs"
    
    id = Column(String, primary_key=True)
    evaluation_id = Column(String, ForeignKey("evaluations.id"))
    step_name = Column(String, nullable=False)  # hard_match, soft_match, llm_evaluation, etc.
    step_status = Column(String, nullable=False)  # success, error, warning
    step_data = Column(JSON)  # Step-specific data
    execution_time = Column(Float)  # Time taken for this step
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ApiKey(Base):
    """API Keys for external services (encrypted)."""
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    service_name = Column(String, nullable=False)  # openai, anthropic, etc.
    encrypted_key = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True))


class SystemSettings(Base):
    """System-wide settings and configurations."""
    __tablename__ = "system_settings"
    
    id = Column(String, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text)
    updated_by = Column(String, ForeignKey("users.id"))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())