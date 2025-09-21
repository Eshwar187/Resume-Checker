from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from typing import List, Optional
import uuid
import logging
from datetime import datetime

from app.schemas import (
    ResumeResponse, BaseResponse, UserResponse, 
    ResumeUploadRequest, FileUploadResponse
)
from app.core.auth import get_current_user, require_role
from app.core.supabase import get_supabase_service_client, get_storage_manager
from app.core.file_processing import get_resume_processor
from app.schemas import UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resume", tags=["Resume Management"])


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_resume(
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(require_role(UserRole.STUDENT))
):
    """
    Upload and process a resume file.
    
    - **file**: Resume file (PDF, DOCX, or TXT)
    
    Returns the processed resume with extracted information.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Generate unique file ID and path
        resume_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1].lower()
        storage_path = f"resumes/{current_user.id}/{resume_id}.{file_extension}"
        
        # Process the resume
        processor = get_resume_processor()
        processed_data = processor.process_resume(file_content, file.filename)
        
        # Upload file to storage
        storage_manager = get_storage_manager()
        file_url = await storage_manager.upload_file(
            file_path=storage_path,
            file_content=file_content,
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Store resume metadata in database
        supabase = get_supabase_service_client()
        resume_data = {
            "id": resume_id,
            "user_id": current_user.id,
            "file_name": file.filename,
            "file_url": file_url,
            "file_size": file_size,
            "file_type": file.content_type or "application/octet-stream",
            "raw_text": processed_data["raw_text"],
            "processed_text": processed_data["processed_text"],
            "extracted_skills": processed_data["extracted_skills"],
            "extracted_experience": processed_data["extracted_experience"]
        }
        
        result = supabase.table("resumes").insert(resume_data).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save resume data"
            )
        
        logger.info(f"Resume uploaded successfully: {resume_id} by user {current_user.id}")
        
        return FileUploadResponse(
            file_id=resume_id,
            file_name=file.filename,
            file_url=file_url,
            file_size=file_size,
            upload_status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume upload failed: {str(e)}"
        )


@router.get("/list", response_model=List[ResumeResponse])
async def list_user_resumes(
    limit: int = 20,
    offset: int = 0,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get list of resumes uploaded by the current user.
    
    - **limit**: Maximum number of resumes to return (default: 20)
    - **offset**: Number of resumes to skip (default: 0)
    """
    try:
        supabase = get_supabase_service_client()
        
        query = supabase.table("resumes").select("*").eq("user_id", current_user.id)
        
        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)
        
        result = query.order("uploaded_at", desc=True).execute()
        
        resumes = [ResumeResponse(**resume) for resume in result.data]
        
        logger.info(f"Retrieved {len(resumes)} resumes for user {current_user.id}")
        return resumes
        
    except Exception as e:
        logger.error(f"Error retrieving resumes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve resumes"
        )


@router.get("/{resume_id}", response_model=ResumeResponse)
async def get_resume(
    resume_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get detailed information about a specific resume.
    
    - **resume_id**: ID of the resume to retrieve
    """
    try:
        supabase = get_supabase_service_client()
        
        result = supabase.table("resumes").select("*").eq("id", resume_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        resume_data = result.data
        
        # Check if user has access to this resume
        if resume_data["user_id"] != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this resume"
            )
        
        return ResumeResponse(**resume_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving resume {resume_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve resume"
        )


@router.delete("/{resume_id}", response_model=BaseResponse)
async def delete_resume(
    resume_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Delete a resume and its associated file.
    
    - **resume_id**: ID of the resume to delete
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get resume data first
        result = supabase.table("resumes").select("*").eq("id", resume_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        resume_data = result.data
        
        # Check if user has access to delete this resume
        if resume_data["user_id"] != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to delete this resume"
            )
        
        # Delete file from storage
        storage_manager = get_storage_manager()
        file_path = resume_data["file_url"].split("/")[-1]  # Extract file path from URL
        storage_path = f"resumes/{resume_data['user_id']}/{file_path}"
        
        try:
            await storage_manager.delete_file(storage_path)
        except Exception as e:
            logger.warning(f"Failed to delete file from storage: {e}")
        
        # Delete resume record from database
        delete_result = supabase.table("resumes").delete().eq("id", resume_id).execute()
        
        if not delete_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete resume"
            )
        
        logger.info(f"Resume deleted successfully: {resume_id} by user {current_user.id}")
        
        return BaseResponse(message="Resume deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resume {resume_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete resume"
        )


@router.put("/{resume_id}/reprocess", response_model=ResumeResponse)
async def reprocess_resume(
    resume_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Reprocess a resume to extract updated information.
    
    - **resume_id**: ID of the resume to reprocess
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get resume data
        result = supabase.table("resumes").select("*").eq("id", resume_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        resume_data = result.data
        
        # Check access
        if resume_data["user_id"] != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this resume"
            )
        
        # Download file from storage
        storage_manager = get_storage_manager()
        file_path = resume_data["file_url"].split("/")[-1]
        storage_path = f"resumes/{resume_data['user_id']}/{file_path}"
        
        file_content = await storage_manager.download_file(storage_path)
        
        # Reprocess the resume
        processor = get_resume_processor()
        processed_data = processor.process_resume(file_content, resume_data["file_name"])
        
        # Update database with new processed data
        update_data = {
            "processed_text": processed_data["processed_text"],
            "extracted_skills": processed_data["extracted_skills"],
            "extracted_experience": processed_data["extracted_experience"]
        }
        
        update_result = supabase.table("resumes").update(update_data).eq("id", resume_id).execute()
        
        if not update_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update resume"
            )
        
        logger.info(f"Resume reprocessed successfully: {resume_id}")
        
        return ResumeResponse(**update_result.data[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing resume {resume_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reprocess resume"
        )