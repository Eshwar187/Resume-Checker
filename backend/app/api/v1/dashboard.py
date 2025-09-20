"""
Dashboard API endpoints for students and placement teams.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from app.schemas import (
    StudentDashboardResponse, PlacementDashboardResponse, UserResponse,
    ResumeResponse, EvaluationResponse, JobDescriptionResponse,
    AnalyticsRequest, AnalyticsResponse
)
from app.core.auth import get_current_user, require_role, require_roles
from app.core.supabase import get_supabase_service_client
from app.schemas import UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/student", response_model=StudentDashboardResponse)
async def get_student_dashboard(
    current_user: UserResponse = Depends(require_role(UserRole.STUDENT))
):
    """
    Get student dashboard with resume and evaluation data.
    
    Returns:
    - User information
    - List of uploaded resumes
    - Evaluation results
    - Summary statistics
    - Recent activity
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get user's resumes
        resumes_result = supabase.table("resumes")\
            .select("*")\
            .eq("user_id", current_user.id)\
            .order("uploaded_at", desc=True)\
            .execute()
        
        resumes = [ResumeResponse(**resume) for resume in resumes_result.data]
        
        # Get evaluations for user's resumes
        resume_ids = [resume.id for resume in resumes]
        evaluations = []
        
        if resume_ids:
            evaluations_result = supabase.table("evaluations")\
                .select("*, job_descriptions!inner(title, company)")\
                .in_("resume_id", resume_ids)\
                .order("created_at", desc=True)\
                .limit(10)\
                .execute()
            
            for eval_data in evaluations_result.data:
                # Clean nested data
                clean_eval = {k: v for k, v in eval_data.items() if k != "job_descriptions"}
                evaluations.append(EvaluationResponse(**clean_eval))
        
        # Calculate statistics
        total_resumes = len(resumes)
        
        scores = [eval.relevance_score for eval in evaluations if eval.relevance_score is not None]
        average_score = sum(scores) / len(scores) if scores else None
        
        # Get recent activity
        recent_activity = []
        
        # Add recent resume uploads
        for resume in resumes[:5]:
            recent_activity.append({
                "type": "resume_upload",
                "description": f"Uploaded resume: {resume.file_name}",
                "timestamp": resume.uploaded_at,
                "data": {"resume_id": resume.id}
            })
        
        # Add recent evaluations
        for evaluation in evaluations[:5]:
            recent_activity.append({
                "type": "evaluation",
                "description": f"Resume evaluated - Score: {evaluation.relevance_score:.1f}",
                "timestamp": evaluation.created_at,
                "data": {
                    "evaluation_id": evaluation.id,
                    "score": evaluation.relevance_score,
                    "verdict": evaluation.verdict
                }
            })
        
        # Sort by timestamp
        recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_activity = recent_activity[:10]
        
        response = StudentDashboardResponse(
            user=current_user,
            resumes=resumes,
            evaluations=evaluations,
            total_resumes=total_resumes,
            average_score=average_score,
            recent_activity=recent_activity
        )
        
        logger.info(f"Student dashboard loaded for user {current_user.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error loading student dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load dashboard"
        )


@router.get("/student/{user_id}", response_model=StudentDashboardResponse)
async def get_student_dashboard_by_id(
    user_id: str,
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM, UserRole.ADMIN]))
):
    """
    Get student dashboard by user ID (for placement teams and admins).
    
    - **user_id**: ID of the student user
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get target user
        user_result = supabase.table("users").select("*").eq("id", user_id).single().execute()
        if not user_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        target_user = UserResponse(**user_result.data)
        
        # Get user's resumes
        resumes_result = supabase.table("resumes")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("uploaded_at", desc=True)\
            .execute()
        
        resumes = [ResumeResponse(**resume) for resume in resumes_result.data]
        
        # Get evaluations
        resume_ids = [resume.id for resume in resumes]
        evaluations = []
        
        if resume_ids:
            evaluations_result = supabase.table("evaluations")\
                .select("*")\
                .in_("resume_id", resume_ids)\
                .order("created_at", desc=True)\
                .execute()
            
            evaluations = [EvaluationResponse(**eval_data) for eval_data in evaluations_result.data]
        
        # Calculate statistics
        total_resumes = len(resumes)
        scores = [eval.relevance_score for eval in evaluations if eval.relevance_score is not None]
        average_score = sum(scores) / len(scores) if scores else None
        
        # Recent activity (simplified for external view)
        recent_activity = []
        for resume in resumes[:5]:
            recent_activity.append({
                "type": "resume_upload",
                "description": f"Uploaded resume: {resume.file_name}",
                "timestamp": resume.uploaded_at,
                "data": {"resume_id": resume.id}
            })
        
        response = StudentDashboardResponse(
            user=target_user,
            resumes=resumes,
            evaluations=evaluations,
            total_resumes=total_resumes,
            average_score=average_score,
            recent_activity=recent_activity
        )
        
        logger.info(f"Student dashboard loaded for user {user_id} by {current_user.id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading student dashboard for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load dashboard"
        )


@router.get("/placement/{jd_id}", response_model=PlacementDashboardResponse)
async def get_placement_dashboard(
    jd_id: str,
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM, UserRole.ADMIN]))
):
    """
    Get placement team dashboard for a specific job description.
    
    - **jd_id**: ID of the job description
    
    Returns:
    - Job description details
    - Candidate evaluation statistics
    - Top candidates
    - Score distribution
    - Recent evaluations
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get job description
        jd_result = supabase.table("job_descriptions").select("*").eq("id", jd_id).single().execute()
        if not jd_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job description not found"
            )
        
        jd_data = jd_result.data
        
        # Check access
        if (jd_data["uploader_id"] != current_user.id and 
            current_user.role != UserRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job description"
            )
        
        job_description = JobDescriptionResponse(**jd_data)
        
        # Get all evaluations for this JD
        evaluations_result = supabase.table("evaluations")\
            .select("*, resumes!inner(user_id, file_name), users!inner(name, email)")\
            .eq("jd_id", jd_id)\
            .order("relevance_score", desc=True)\
            .execute()
        
        all_evaluations = evaluations_result.data
        
        # Calculate statistics
        total_candidates = len(set(eval["resumes"]["user_id"] for eval in all_evaluations))
        evaluated_candidates = len(all_evaluations)
        
        scores = [eval["relevance_score"] for eval in all_evaluations if eval["relevance_score"] is not None]
        average_score = sum(scores) / len(scores) if scores else None
        
        # Score distribution
        score_distribution = {"High": 0, "Medium": 0, "Low": 0}
        for eval in all_evaluations:
            verdict = eval.get("verdict", "Low")
            score_distribution[verdict] = score_distribution.get(verdict, 0) + 1
        
        # Top candidates (top 10 by score)
        top_candidates = []
        seen_users = set()
        
        for eval in all_evaluations[:20]:  # Check more evaluations to get diverse candidates
            user_id = eval["resumes"]["user_id"]
            if user_id not in seen_users:
                top_candidates.append({
                    "user_id": user_id,
                    "user_name": eval["users"]["name"],
                    "user_email": eval["users"]["email"],
                    "resume_file": eval["resumes"]["file_name"],
                    "evaluation_id": eval["id"],
                    "relevance_score": eval["relevance_score"],
                    "verdict": eval["verdict"],
                    "matched_skills": eval["matched_skills"],
                    "missing_skills": eval["missing_skills"],
                    "created_at": eval["created_at"]
                })
                seen_users.add(user_id)
                
                if len(top_candidates) >= 10:
                    break
        
        # Recent evaluations (last 10)
        recent_evaluations = []
        for eval_data in all_evaluations[:10]:
            clean_eval = {k: v for k, v in eval_data.items() 
                         if k not in ["resumes", "users"]}
            recent_evaluations.append(EvaluationResponse(**clean_eval))
        
        response = PlacementDashboardResponse(
            job_description=job_description,
            total_candidates=total_candidates,
            evaluated_candidates=evaluated_candidates,
            average_score=average_score,
            score_distribution=score_distribution,
            top_candidates=top_candidates,
            recent_evaluations=recent_evaluations
        )
        
        logger.info(f"Placement dashboard loaded for JD {jd_id} by {current_user.id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading placement dashboard for JD {jd_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load dashboard"
        )


@router.get("/placement", response_model=List[Dict[str, Any]])
async def get_placement_overview(
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM, UserRole.ADMIN]))
):
    """
    Get overview of all job descriptions for placement team.
    
    Returns list of job descriptions with basic statistics.
    """
    try:
        supabase = get_supabase_service_client()
        
        # Get user's job descriptions
        jds_query = supabase.table("job_descriptions").select("*")
        
        # Filter by uploader for placement team members
        if current_user.role == UserRole.PLACEMENT_TEAM:
            jds_query = jds_query.eq("uploader_id", current_user.id)
        
        jds_result = jds_query.order("created_at", desc=True).execute()
        
        overview = []
        
        for jd in jds_result.data:
            jd_id = jd["id"]
            
            # Get evaluation count for this JD
            eval_count_result = supabase.table("evaluations")\
                .select("id", count="exact")\
                .eq("jd_id", jd_id)\
                .execute()
            
            evaluation_count = eval_count_result.count or 0
            
            # Get average score
            if evaluation_count > 0:
                eval_scores_result = supabase.table("evaluations")\
                    .select("relevance_score")\
                    .eq("jd_id", jd_id)\
                    .execute()
                
                scores = [eval["relevance_score"] for eval in eval_scores_result.data 
                         if eval["relevance_score"] is not None]
                average_score = sum(scores) / len(scores) if scores else None
            else:
                average_score = None
            
            overview.append({
                "jd_id": jd_id,
                "title": jd["title"],
                "company": jd["company"],
                "created_at": jd["created_at"],
                "evaluation_count": evaluation_count,
                "average_score": average_score,
                "must_have_skills": jd.get("must_have_skills", []),
                "location": jd.get("location", ""),
                "job_type": jd.get("job_type", "")
            })
        
        logger.info(f"Placement overview loaded for user {current_user.id} - {len(overview)} JDs")
        return overview
        
    except Exception as e:
        logger.error(f"Error loading placement overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load overview"
        )


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    analytics_request: AnalyticsRequest = Depends(),
    current_user: UserResponse = Depends(require_roles([UserRole.PLACEMENT_TEAM, UserRole.ADMIN]))
):
    """
    Get analytics data for evaluations.
    
    - **start_date**: Start date for analytics (optional)
    - **end_date**: End date for analytics (optional)
    - **user_id**: Filter by specific user (optional)
    - **jd_id**: Filter by specific job description (optional)
    """
    try:
        supabase = get_supabase_service_client()
        
        # Build base query
        query = supabase.table("evaluations").select("*")
        
        # Apply filters
        if analytics_request.start_date:
            query = query.gte("created_at", analytics_request.start_date.isoformat())
        
        if analytics_request.end_date:
            query = query.lte("created_at", analytics_request.end_date.isoformat())
        
        if analytics_request.user_id:
            # Filter by user's resumes
            user_resumes = supabase.table("resumes")\
                .select("id")\
                .eq("user_id", analytics_request.user_id)\
                .execute()
            
            resume_ids = [r["id"] for r in user_resumes.data]
            if resume_ids:
                query = query.in_("resume_id", resume_ids)
            else:
                # No resumes for this user
                return AnalyticsResponse(
                    total_evaluations=0,
                    average_processing_time=0.0,
                    score_distribution={},
                    popular_skills=[],
                    trends={}
                )
        
        if analytics_request.jd_id:
            query = query.eq("jd_id", analytics_request.jd_id)
        
        # For placement team members, filter by their JDs
        if current_user.role == UserRole.PLACEMENT_TEAM and not analytics_request.jd_id:
            user_jds = supabase.table("job_descriptions")\
                .select("id")\
                .eq("uploader_id", current_user.id)\
                .execute()
            
            jd_ids = [jd["id"] for jd in user_jds.data]
            if jd_ids:
                query = query.in_("jd_id", jd_ids)
            else:
                # No JDs for this user
                return AnalyticsResponse(
                    total_evaluations=0,
                    average_processing_time=0.0,
                    score_distribution={},
                    popular_skills=[],
                    trends={}
                )
        
        # Execute query
        result = query.execute()
        evaluations = result.data
        
        # Calculate analytics
        total_evaluations = len(evaluations)
        
        # Average processing time
        processing_times = [eval["processing_time"] for eval in evaluations 
                          if eval.get("processing_time") is not None]
        average_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # Score distribution
        score_distribution = {"High": 0, "Medium": 0, "Low": 0}
        for eval in evaluations:
            verdict = eval.get("verdict", "Low")
            score_distribution[verdict] = score_distribution.get(verdict, 0) + 1
        
        # Popular skills (from matched skills)
        skill_counts = {}
        for eval in evaluations:
            matched_skills = eval.get("matched_skills", [])
            for skill in matched_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        popular_skills = [
            {"skill": skill, "count": count}
            for skill, count in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Trends (by month)
        trends = {}
        if evaluations:
            # Group by month
            monthly_counts = {}
            monthly_scores = {}
            
            for eval in evaluations:
                created_at = datetime.fromisoformat(eval["created_at"].replace("Z", "+00:00"))
                month_key = created_at.strftime("%Y-%m")
                
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                
                if eval.get("relevance_score") is not None:
                    if month_key not in monthly_scores:
                        monthly_scores[month_key] = []
                    monthly_scores[month_key].append(eval["relevance_score"])
            
            trends = {
                "monthly_evaluations": monthly_counts,
                "monthly_average_scores": {
                    month: sum(scores) / len(scores) if scores else 0
                    for month, scores in monthly_scores.items()
                }
            }
        
        response = AnalyticsResponse(
            total_evaluations=total_evaluations,
            average_processing_time=average_processing_time,
            score_distribution=score_distribution,
            popular_skills=popular_skills,
            trends=trends
        )
        
        logger.info(f"Analytics loaded for user {current_user.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error loading analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load analytics"
        )