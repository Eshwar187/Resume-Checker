from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from typing import List, Optional
import uuid
import logging
from datetime import datetime

from app.schemas import (
    EvaluationRequest, EvaluationResponse, BaseResponse, UserResponse,
    CandidateFilter, SearchRequest, SearchResponse
)
from app.core.auth import get_current_user, require_roles
from app.core.supabase import get_supabase_service_client
from app.core.evaluation import get_evaluation_engine
from app.schemas import UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluate", tags=["Resume Evaluation"])


@router.post("/", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def evaluate_resume_relevance(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Evaluate resume relevance against a job description.
    
    - **resume_id**: ID of the resume to evaluate
    - **jd_id**: ID of the job description to compare against
    - **use_advanced_ai**: Whether to use advanced AI analysis (default: true)
    
    Returns detailed evaluation results including:
    - Relevance score (0-100)
    - Verdict (High/Medium/Low)
    - Matched and missing skills
    - Experience analysis
    - Personalized feedback and recommendations
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get resume data
        resume_result = supabase.table("resumes").select("*").eq("id", request.resume_id).single().execute()
        if not resume_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        resume_data = resume_result.data
        
        # Check if user has access to this resume
        if (resume_data["user_id"] != current_user.id and 
            current_user.role not in [UserRole.PLACEMENT_TEAM, UserRole.ADMIN]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this resume"
            )
        
        # Get job description data
        jd_result = supabase.table("job_descriptions").select("*").eq("id", request.jd_id).single().execute()
        if not jd_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job description not found"
            )
        
        jd_data = jd_result.data
        
        # Check if evaluation already exists
        existing_eval = supabase.table("evaluations").select("*").eq("resume_id", request.resume_id).eq("jd_id", request.jd_id).execute()
        
        if existing_eval.data:
            logger.info(f"Returning existing evaluation for resume {request.resume_id} and JD {request.jd_id}")
            return EvaluationResponse(**existing_eval.data[0])
        
        # Perform evaluation
        logger.info(f"Starting evaluation for resume {request.resume_id} against JD {request.jd_id}")
        
        evaluation_engine = get_evaluation_engine()
        evaluation_result = await evaluation_engine.evaluate_resume(
            resume_data=resume_data,
            jd_data=jd_data,
            use_advanced_ai=request.use_advanced_ai
        )
        
        # Save evaluation results
        evaluation_id = str(uuid.uuid4())
        evaluation_data = {
            "id": evaluation_id,
            "resume_id": request.resume_id,
            "jd_id": request.jd_id,
            "relevance_score": evaluation_result["relevance_score"],
            "verdict": evaluation_result["verdict"],
            "matched_skills": evaluation_result["matched_skills"],
            "missing_skills": evaluation_result["missing_skills"],
            "skill_match_percentage": evaluation_result["skill_match_percentage"],
            "experience_match": evaluation_result["experience_match"],
            "experience_gap": evaluation_result["experience_gap"],
            "relevant_projects": evaluation_result["relevant_projects"],
            "missing_projects": evaluation_result["missing_projects"],
            "feedback": evaluation_result["feedback"],
            "recommendations": evaluation_result["recommendations"],
            "hard_match_score": evaluation_result["hard_match_score"],
            "soft_match_score": evaluation_result["soft_match_score"],
            "processing_time": evaluation_result["processing_time"],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("evaluations").insert(evaluation_data).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save evaluation results"
            )
        
        # Log evaluation steps in background
        background_tasks.add_task(
            log_evaluation_steps,
            evaluation_id,
            evaluation_result.get("processing_logs", [])
        )
        
        logger.info(f"Evaluation completed successfully: {evaluation_id}")
        
        return EvaluationResponse(**result.data[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get detailed evaluation results.
    
    - **evaluation_id**: ID of the evaluation to retrieve
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get evaluation with resume and JD info for access control
        result = supabase.table("evaluations")\
            .select("*, resumes!inner(user_id), job_descriptions!inner(uploader_id)")\
            .eq("id", evaluation_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        evaluation_data = result.data
        resume_user_id = evaluation_data["resumes"]["user_id"]
        jd_uploader_id = evaluation_data["job_descriptions"]["uploader_id"]
        
        # Check access
        has_access = (
            current_user.id == resume_user_id or  # Resume owner
            current_user.id == jd_uploader_id or  # JD uploader
            current_user.role == UserRole.ADMIN  # Admin
        )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this evaluation"
            )
        
        # Remove nested objects for response
        clean_data = {k: v for k, v in evaluation_data.items() 
                     if k not in ["resumes", "job_descriptions"]}
        
        return EvaluationResponse(**clean_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluation {evaluation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluation"
        )


@router.get("/resume/{resume_id}", response_model=List[EvaluationResponse])
async def get_resume_evaluations(
    resume_id: str,
    limit: int = 20,
    offset: int = 0,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get all evaluations for a specific resume.
    
    - **resume_id**: ID of the resume
    - **limit**: Maximum number of evaluations to return
    - **offset**: Number of evaluations to skip
    """
    try:
        supabase = get_supabase_service_client()
        
        # Check if user has access to this resume
        resume_result = supabase.table("resumes").select("user_id").eq("id", resume_id).single().execute()
        if not resume_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        if (resume_result.data["user_id"] != current_user.id and 
            current_user.role != UserRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this resume"
            )
        
        # Get evaluations
        query = supabase.table("evaluations").select("*").eq("resume_id", resume_id)
        
        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)
        
        result = query.order("created_at", desc=True).execute()
        
        evaluations = [EvaluationResponse(**eval_data) for eval_data in result.data]
        
        logger.info(f"Retrieved {len(evaluations)} evaluations for resume {resume_id}")
        return evaluations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluations for resume {resume_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluations"
        )


@router.get("/jd/{jd_id}", response_model=List[EvaluationResponse])
async def get_jd_evaluations(
    jd_id: str,
    filters: CandidateFilter = Depends(),
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM]))
):
    """
    Get all evaluations for a specific job description with filtering.
    
    - **jd_id**: ID of the job description
    - **filters**: Filtering options (score range, verdict, skills, etc.)
    """
    try:
        supabase = get_supabase_service_client()
        
        # Check if user has access to this JD
        jd_result = supabase.table("job_descriptions").select("uploader_id").eq("id", jd_id).single().execute()
        if not jd_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job description not found"
            )
        
        if (jd_result.data["uploader_id"] != current_user.id and 
            current_user.role != UserRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job description"
            )
        
        # Build query with filters
        query = supabase.table("evaluations").select("*").eq("jd_id", jd_id)
        
        # Apply filters
        if filters.min_score is not None:
            query = query.gte("relevance_score", filters.min_score)
        
        if filters.max_score is not None:
            query = query.lte("relevance_score", filters.max_score)
        
        if filters.verdict:
            verdict_values = [v.value for v in filters.verdict]
            query = query.in_("verdict", verdict_values)
        
        if filters.experience_match is not None:
            query = query.eq("experience_match", filters.experience_match)
        
        # Apply pagination
        if filters.limit:
            query = query.limit(filters.limit)
        if filters.offset:
            query = query.offset(filters.offset)
        
        result = query.order("relevance_score", desc=True).execute()
        
        evaluations = [EvaluationResponse(**eval_data) for eval_data in result.data]
        
        # Filter by required skills if specified
        if filters.required_skills:
            filtered_evaluations = []
            for evaluation in evaluations:
                matched_skills_lower = [skill.lower() for skill in evaluation.matched_skills]
                required_skills_lower = [skill.lower() for skill in filters.required_skills]
                
                # Check if all required skills are matched
                if all(skill in matched_skills_lower for skill in required_skills_lower):
                    filtered_evaluations.append(evaluation)
            
            evaluations = filtered_evaluations
        
        logger.info(f"Retrieved {len(evaluations)} evaluations for JD {jd_id}")
        return evaluations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluations for JD {jd_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluations"
        )


@router.delete("/{evaluation_id}", response_model=BaseResponse)
async def delete_evaluation(
    evaluation_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Delete an evaluation.
    
    - **evaluation_id**: ID of the evaluation to delete
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get evaluation with access control info
        result = supabase.table("evaluations")\
            .select("*, resumes!inner(user_id), job_descriptions!inner(uploader_id)")\
            .eq("id", evaluation_id).single().execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        evaluation_data = result.data
        resume_user_id = evaluation_data["resumes"]["user_id"]
        jd_uploader_id = evaluation_data["job_descriptions"]["uploader_id"]
        
        # Check access
        has_access = (
            current_user.id == resume_user_id or  # Resume owner
            current_user.id == jd_uploader_id or  # JD uploader
            current_user.role == UserRole.ADMIN  # Admin
        )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to delete this evaluation"
            )
        
        # Delete evaluation
        delete_result = supabase.table("evaluations").delete().eq("id", evaluation_id).execute()
        
        if not delete_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete evaluation"
            )
        
        logger.info(f"Evaluation deleted successfully: {evaluation_id}")
        
        return BaseResponse(message="Evaluation deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting evaluation {evaluation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete evaluation"
        )


async def log_evaluation_steps(evaluation_id: str, step_logs: List[dict]):
    """Background task to log evaluation steps."""
    try:
        supabase = get_supabase_service_client()
        
        for step_log in step_logs:
            log_data = {
                "id": str(uuid.uuid4()),
                "evaluation_id": evaluation_id,
                "step_name": step_log.get("step", "unknown"),
                "step_status": "success",
                "step_data": step_log,
                "execution_time": step_log.get("execution_time", 0.0)
            }
            
            supabase.table("evaluation_logs").insert(log_data).execute()
        
        logger.info(f"Logged {len(step_logs)} evaluation steps for {evaluation_id}")
        
    except Exception as e:
        logger.error(f"Failed to log evaluation steps: {e}")