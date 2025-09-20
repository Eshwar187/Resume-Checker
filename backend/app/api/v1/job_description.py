"""
Job Description upload and management API endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from typing import List, Optional
import uuid
import logging

from app.schemas import (
    JobDescriptionResponse, BaseResponse, UserResponse,
    JobDescriptionUploadRequest, FileUploadResponse
)
from app.core.auth import get_current_user, require_roles
from app.core.supabase import get_supabase_service_client, get_storage_manager
from app.core.file_processing import get_job_description_processor
from app.schemas import UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jd", tags=["Job Description Management"])


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_job_description(
    title: str = Form(...),
    company: Optional[str] = Form(None),
    job_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM]))
):
    """
    Upload and process a job description.
    
    - **title**: Job title
    - **company**: Company name (optional)
    - **job_text**: Job description text (optional, if not uploading file)
    - **file**: Job description file (PDF, DOCX, or TXT) (optional)
    
    Either job_text or file must be provided.
    """
    try:
        # Validate input
        if not job_text and not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either job_text or file must be provided"
            )
        
        jd_id = str(uuid.uuid4())
        file_content = b""
        file_url = None
        file_name = None
        file_size = 0
        
        # Handle file upload if provided
        if file and file.filename:
            file_content = await file.read()
            file_size = len(file_content)
            
            # Generate storage path
            file_extension = file.filename.split('.')[-1].lower()
            storage_path = f"job_descriptions/{current_user.id}/{jd_id}.{file_extension}"
            
            # Upload file to storage
            storage_manager = get_storage_manager()
            file_url = await storage_manager.upload_file(
                file_path=storage_path,
                file_content=file_content,
                content_type=file.content_type or "application/octet-stream"
            )
            file_name = file.filename
        
        # Process the job description
        processor = get_job_description_processor()
        
        if job_text and not file_content:
            # Use provided text directly
            processed_data = processor.process_job_description(b"", "", title)
            processed_data["raw_text"] = job_text
            processed_data["processed_text"] = processor._clean_text(job_text)
            # Re-extract information from the provided text
            processed_data.update({
                "job_title": processor._extract_job_title(job_text, title),
                "must_have_skills": processor._extract_required_skills(job_text),
                "good_to_have_skills": processor._extract_preferred_skills(job_text),
                "qualifications": processor._extract_qualifications(job_text),
                "experience_required": processor._extract_experience_requirements(job_text),
                "location": processor._extract_location(job_text),
                "job_type": processor._extract_job_type(job_text)
            })
        else:
            # Process uploaded file
            processed_data = processor.process_job_description(file_content, file_name or "", title)
        
        # Store job description in database
        supabase = get_supabase_service_client()
        jd_data = {
            "id": jd_id,
            "uploader_id": current_user.id,
            "title": title,
            "company": company,
            "file_name": file_name,
            "file_url": file_url,
            "raw_text": processed_data["raw_text"],
            "processed_text": processed_data["processed_text"],
            "job_title": processed_data["job_title"],
            "must_have_skills": processed_data["must_have_skills"],
            "good_to_have_skills": processed_data["good_to_have_skills"],
            "qualifications": processed_data["qualifications"],
            "experience_required": processed_data["experience_required"],
            "location": processed_data["location"],
            "job_type": processed_data["job_type"]
        }
        
        result = supabase.table("job_descriptions").insert(jd_data).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save job description"
            )
        
        logger.info(f"Job description uploaded successfully: {jd_id} by user {current_user.id}")
        
        return FileUploadResponse(
            file_id=jd_id,
            file_name=file_name or f"{title}.txt",
            file_url=file_url or "",
            file_size=file_size,
            upload_status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job description upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job description upload failed: {str(e)}"
        )


@router.get("/list", response_model=List[JobDescriptionResponse])
async def list_job_descriptions(
    limit: int = 20,
    offset: int = 0,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get list of job descriptions.
    
    - **limit**: Maximum number of job descriptions to return (default: 20)
    - **offset**: Number of job descriptions to skip (default: 0)
    
    Students see all job descriptions, placement team members see only their own.
    """
    try:
        supabase = get_supabase_service_client()
        
        query = supabase.table("job_descriptions").select("*")
        
        # Filter by uploader for placement team members (not admins)
        if current_user.role == UserRole.PLACEMENT_TEAM:
            query = query.eq("uploader_id", current_user.id)
        
        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)
        
        result = query.order("created_at", desc=True).execute()
        
        job_descriptions = [JobDescriptionResponse(**jd) for jd in result.data]
        
        logger.info(f"Retrieved {len(job_descriptions)} job descriptions for user {current_user.id}")
        return job_descriptions
        
    except Exception as e:
        logger.error(f"Error retrieving job descriptions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job descriptions"
        )


@router.get("/{jd_id}", response_model=JobDescriptionResponse)
async def get_job_description(
    jd_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get detailed information about a specific job description.
    
    - **jd_id**: ID of the job description to retrieve
    """
    try:
        supabase = get_supabase_service_client()
        
        result = supabase.table("job_descriptions").select("*").eq("id", jd_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job description not found"
            )
        
        jd_data = result.data
        
        # Check access for placement team members
        if (current_user.role == UserRole.PLACEMENT_TEAM and 
            jd_data["uploader_id"] != current_user.id and 
            current_user.role != UserRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job description"
            )
        
        return JobDescriptionResponse(**jd_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job description {jd_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job description"
        )


@router.put("/{jd_id}", response_model=JobDescriptionResponse)
async def update_job_description(
    jd_id: str,
    title: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    job_text: Optional[str] = Form(None),
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM]))
):
    """
    Update a job description.
    
    - **jd_id**: ID of the job description to update
    - **title**: New job title (optional)
    - **company**: New company name (optional)
    - **job_text**: New job description text (optional)
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get existing job description
        result = supabase.table("job_descriptions").select("*").eq("id", jd_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job description not found"
            )
        
        jd_data = result.data
        
        # Check access
        if jd_data["uploader_id"] != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to update this job description"
            )
        
        # Prepare update data
        update_data = {}
        
        if title is not None:
            update_data["title"] = title
        if company is not None:
            update_data["company"] = company
        
        # If job_text is provided, reprocess the content
        if job_text is not None:
            processor = get_job_description_processor()
            processed_data = processor.process_job_description(b"", "", title or jd_data["title"])
            processed_data["raw_text"] = job_text
            processed_data["processed_text"] = processor._clean_text(job_text)
            
            # Re-extract information
            processed_data.update({
                "job_title": processor._extract_job_title(job_text, title or jd_data["title"]),
                "must_have_skills": processor._extract_required_skills(job_text),
                "good_to_have_skills": processor._extract_preferred_skills(job_text),
                "qualifications": processor._extract_qualifications(job_text),
                "experience_required": processor._extract_experience_requirements(job_text),
                "location": processor._extract_location(job_text),
                "job_type": processor._extract_job_type(job_text)
            })
            
            update_data.update({
                "raw_text": processed_data["raw_text"],
                "processed_text": processed_data["processed_text"],
                "job_title": processed_data["job_title"],
                "must_have_skills": processed_data["must_have_skills"],
                "good_to_have_skills": processed_data["good_to_have_skills"],
                "qualifications": processed_data["qualifications"],
                "experience_required": processed_data["experience_required"],
                "location": processed_data["location"],
                "job_type": processed_data["job_type"]
            })
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
        
        # Update in database
        update_result = supabase.table("job_descriptions").update(update_data).eq("id", jd_id).execute()
        
        if not update_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update job description"
            )
        
        logger.info(f"Job description updated successfully: {jd_id}")
        
        return JobDescriptionResponse(**update_result.data[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job description {jd_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update job description"
        )


@router.delete("/{jd_id}", response_model=BaseResponse)
async def delete_job_description(
    jd_id: str,
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM]))
):
    """
    Delete a job description and its associated file.
    
    - **jd_id**: ID of the job description to delete
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get job description data first
        result = supabase.table("job_descriptions").select("*").eq("id", jd_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job description not found"
            )
        
        jd_data = result.data
        
        # Check access
        if jd_data["uploader_id"] != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to delete this job description"
            )
        
        # Delete file from storage if exists
        if jd_data["file_url"]:
            storage_manager = get_storage_manager()
            file_path = jd_data["file_url"].split("/")[-1]
            storage_path = f"job_descriptions/{jd_data['uploader_id']}/{file_path}"
            
            try:
                await storage_manager.delete_file(storage_path)
            except Exception as e:
                logger.warning(f"Failed to delete file from storage: {e}")
        
        # Delete job description record from database
        delete_result = supabase.table("job_descriptions").delete().eq("id", jd_id).execute()
        
        if not delete_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete job description"
            )
        
        logger.info(f"Job description deleted successfully: {jd_id} by user {current_user.id}")
        
        return BaseResponse(message="Job description deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job description {jd_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete job description"
        )