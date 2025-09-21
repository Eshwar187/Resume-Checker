"""
AI Suggestions API endpoint for real-time resume improvement recommendations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from datetime import datetime

from app.core.auth import get_current_user
from app.schemas import UserResponse
from app.core.evaluation import get_evaluation_engine
from app.core.observability import trace_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-suggestions", tags=["AI Suggestions"])


@router.post("/analyze-resume")
@trace_evaluation
async def analyze_resume_for_suggestions(
    resume_text: str,
    target_role: Optional[str] = None,
    current_user: UserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analyze resume and provide AI-powered improvement suggestions.
    
    Args:
        resume_text: The resume text content
        target_role: Optional target job role for tailored suggestions
        current_user: Authenticated user
    
    Returns:
        Comprehensive AI suggestions for resume improvement
    """
    try:
        evaluation_engine = get_evaluation_engine()
        
        # Generate AI suggestions using LLM
        suggestions = await _generate_ai_suggestions(
            evaluation_engine, resume_text, target_role
        )
        
        return {
            "status": "success",
            "suggestions": suggestions,
            "target_role": target_role,
            "analyzed_at": datetime.utcnow().isoformat(),
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"AI suggestions analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate AI suggestions: {str(e)}"
        )


@router.post("/stream-suggestions")
@trace_evaluation
async def stream_ai_suggestions(
    resume_text: str,
    target_role: Optional[str] = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Stream real-time AI suggestions as they are generated.
    
    This endpoint provides a real-time streaming response for AI suggestions,
    allowing the frontend to display suggestions as they are generated.
    """
    
    async def generate_suggestions_stream():
        """Generate streaming AI suggestions."""
        try:
            evaluation_engine = get_evaluation_engine()
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting AI analysis...'})}\n\n"
            
            # Analyze resume step by step
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting skills and experience...'})}\n\n"
            
            # Extract skills
            skills = evaluation_engine.skill_matcher.nlp_processor.extract_skills_advanced(resume_text)
            yield f"data: {json.dumps({'type': 'skills_extracted', 'skills': skills[:10]})}\n\n"
            
            await asyncio.sleep(0.5)  # Small delay for UX
            
            # Generate suggestions sections
            sections = [
                "skill_recommendations",
                "content_improvements", 
                "formatting_suggestions",
                "keyword_optimization",
                "ats_optimization"
            ]
            
            for section in sections:
                yield f"data: {json.dumps({'type': 'status', 'message': f'Generating {section.replace(\"_\", \" \")}...'})}\n\n"
                
                section_suggestions = await _generate_section_suggestions(
                    evaluation_engine, resume_text, section, target_role, skills
                )
                
                yield f"data: {json.dumps({'type': section, 'suggestions': section_suggestions})}\n\n"
                await asyncio.sleep(0.3)  # Small delay between sections
            
            # Final summary
            yield f"data: {json.dumps({'type': 'complete', 'message': 'AI analysis complete!'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming suggestions failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_suggestions_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )


async def _generate_ai_suggestions(
    evaluation_engine, 
    resume_text: str, 
    target_role: Optional[str]
) -> Dict[str, Any]:
    """Generate comprehensive AI suggestions."""
    
    # Extract current skills
    skills = evaluation_engine.skill_matcher.nlp_processor.extract_skills_advanced(resume_text)
    
    # Create suggestions structure
    suggestions = {
        "skill_recommendations": await _generate_skill_recommendations(evaluation_engine, resume_text, skills, target_role),
        "content_improvements": await _generate_content_improvements(evaluation_engine, resume_text, target_role),
        "formatting_suggestions": _generate_formatting_suggestions(resume_text),
        "keyword_optimization": await _generate_keyword_optimization(evaluation_engine, resume_text, target_role),
        "ats_optimization": _generate_ats_optimization(resume_text, skills),
        "overall_score": _calculate_overall_score(resume_text, skills),
        "priority_actions": await _generate_priority_actions(evaluation_engine, resume_text, skills, target_role)
    }
    
    return suggestions


async def _generate_section_suggestions(
    evaluation_engine,
    resume_text: str,
    section: str,
    target_role: Optional[str],
    skills: List[str]
) -> List[Dict[str, Any]]:
    """Generate suggestions for a specific section."""
    
    if section == "skill_recommendations":
        return await _generate_skill_recommendations(evaluation_engine, resume_text, skills, target_role)
    elif section == "content_improvements":
        return await _generate_content_improvements(evaluation_engine, resume_text, target_role)
    elif section == "formatting_suggestions":
        return _generate_formatting_suggestions(resume_text)
    elif section == "keyword_optimization":
        return await _generate_keyword_optimization(evaluation_engine, resume_text, target_role)
    elif section == "ats_optimization":
        return _generate_ats_optimization(resume_text, skills)
    else:
        return []


async def _generate_skill_recommendations(
    evaluation_engine, 
    resume_text: str, 
    current_skills: List[str], 
    target_role: Optional[str]
) -> List[Dict[str, Any]]:
    """Generate skill-based recommendations using AI."""
    
    # Define trending skills by role
    trending_skills = {
        "software engineer": ["React", "TypeScript", "Docker", "Kubernetes", "AWS", "GraphQL", "Next.js"],
        "data scientist": ["PyTorch", "MLflow", "Streamlit", "dbt", "Snowflake", "Apache Spark", "TensorFlow"],
        "product manager": ["Figma", "Mixpanel", "A/B Testing", "SQL", "Tableau", "Jira", "Slack"],
        "devops engineer": ["Terraform", "Ansible", "Jenkins", "GitLab CI", "Prometheus", "Grafana", "Helm"],
        "frontend developer": ["React", "Vue.js", "TypeScript", "Tailwind CSS", "Next.js", "Vite", "Storybook"],
        "backend developer": ["Node.js", "FastAPI", "PostgreSQL", "Redis", "Docker", "MongoDB", "Express"],
        "default": ["Python", "SQL", "Git", "Docker", "AWS", "React", "Node.js"]
    }
    
    role_key = target_role.lower() if target_role else "default"
    relevant_skills = trending_skills.get(role_key, trending_skills["default"])
    
    # Find missing trending skills
    missing_skills = [skill for skill in relevant_skills if skill.lower() not in [s.lower() for s in current_skills]]
    
    recommendations = []
    
    for skill in missing_skills[:5]:  # Top 5 recommendations
        recommendations.append({
            "type": "skill_gap",
            "title": f"Add {skill} to your skillset",
            "description": f"This skill is highly valued for {target_role or 'your target role'}",
            "priority": "high" if skill in relevant_skills[:3] else "medium",
            "action": f"Consider adding {skill} experience through projects or courses",
            "impact": "High - This skill appears in 80%+ of relevant job postings"
        })
    
    # Skills to emphasize
    strong_skills = [skill for skill in current_skills if skill.lower() in [s.lower() for s in relevant_skills]]
    
    for skill in strong_skills[:3]:
        recommendations.append({
            "type": "skill_emphasis",
            "title": f"Emphasize your {skill} experience",
            "description": f"You have {skill} skills - make them more prominent",
            "priority": "medium",
            "action": f"Add specific {skill} projects or quantify your {skill} experience",
            "impact": "Medium - Better highlight existing valuable skills"
        })
    
    return recommendations


async def _generate_content_improvements(
    evaluation_engine, 
    resume_text: str, 
    target_role: Optional[str]
) -> List[Dict[str, Any]]:
    """Generate content improvement suggestions using LLM."""
    
    suggestions = []
    
    # Check for quantifiable achievements
    if not any(char.isdigit() for char in resume_text):
        suggestions.append({
            "type": "quantification",
            "title": "Add quantifiable achievements",
            "description": "Include numbers, percentages, or metrics to demonstrate impact",
            "priority": "high",
            "action": "Replace vague statements with specific metrics (e.g., 'Improved performance by 30%')",
            "impact": "High - Quantified achievements are 40% more likely to get attention"
        })
    
    # Check for action verbs
    weak_verbs = ["responsible for", "worked on", "involved in", "helped with"]
    if any(verb in resume_text.lower() for verb in weak_verbs):
        suggestions.append({
            "type": "action_verbs",
            "title": "Use stronger action verbs",
            "description": "Replace passive language with powerful action verbs",
            "priority": "medium",
            "action": "Use verbs like 'Developed', 'Implemented', 'Optimized', 'Led', 'Created'",
            "impact": "Medium - Strong verbs make your contributions clearer"
        })
    
    # Check resume length
    word_count = len(resume_text.split())
    if word_count < 300:
        suggestions.append({
            "type": "length",
            "title": "Expand your resume content",
            "description": f"Your resume has {word_count} words - consider adding more detail",
            "priority": "medium",
            "action": "Add more projects, experiences, or detailed descriptions",
            "impact": "Medium - More comprehensive resumes perform better"
        })
    elif word_count > 800:
        suggestions.append({
            "type": "length",
            "title": "Consider condensing your resume",
            "description": f"Your resume has {word_count} words - may be too long",
            "priority": "low",
            "action": "Focus on most relevant experiences and achievements",
            "impact": "Low - Concise resumes are easier to scan"
        })
    
    return suggestions


def _generate_formatting_suggestions(resume_text: str) -> List[Dict[str, Any]]:
    """Generate formatting improvement suggestions."""
    
    suggestions = []
    
    # Check for contact information
    has_email = "@" in resume_text
    has_phone = any(char.isdigit() for char in resume_text)
    
    if not has_email:
        suggestions.append({
            "type": "contact",
            "title": "Add email address",
            "description": "Include a professional email address for easy contact",
            "priority": "high",
            "action": "Add your email at the top of your resume",
            "impact": "Critical - Recruiters need to contact you"
        })
    
    if not has_phone:
        suggestions.append({
            "type": "contact", 
            "title": "Add phone number",
            "description": "Include a phone number for direct contact",
            "priority": "high",
            "action": "Add your phone number in the contact section",
            "impact": "High - Multiple contact methods increase response rates"
        })
    
    # Check for sections
    sections = ["experience", "education", "skills", "projects"]
    missing_sections = []
    
    for section in sections:
        if section.lower() not in resume_text.lower():
            missing_sections.append(section)
    
    if missing_sections:
        suggestions.append({
            "type": "sections",
            "title": f"Consider adding {', '.join(missing_sections)} section(s)",
            "description": "Standard resume sections help recruiters find information quickly",
            "priority": "medium",
            "action": f"Add dedicated sections for {', '.join(missing_sections)}",
            "impact": "Medium - Well-structured resumes are easier to scan"
        })
    
    return suggestions


async def _generate_keyword_optimization(
    evaluation_engine, 
    resume_text: str, 
    target_role: Optional[str]
) -> List[Dict[str, Any]]:
    """Generate keyword optimization suggestions."""
    
    suggestions = []
    
    # Role-specific keywords
    role_keywords = {
        "software engineer": ["agile", "scrum", "ci/cd", "testing", "debugging", "scalable", "architecture"],
        "data scientist": ["analytics", "modeling", "statistics", "visualization", "insights", "algorithms"],
        "product manager": ["roadmap", "stakeholders", "metrics", "user experience", "strategy", "launch"],
        "default": ["collaboration", "leadership", "problem-solving", "innovation", "optimization"]
    }
    
    role_key = target_role.lower() if target_role else "default"
    keywords = role_keywords.get(role_key, role_keywords["default"])
    
    missing_keywords = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
    
    if missing_keywords:
        suggestions.append({
            "type": "keywords",
            "title": "Add industry-relevant keywords",
            "description": f"Include keywords like: {', '.join(missing_keywords[:3])}",
            "priority": "medium",
            "action": "Naturally incorporate these keywords into your experience descriptions",
            "impact": "Medium - ATS systems look for these keywords"
        })
    
    return suggestions


def _generate_ats_optimization(resume_text: str, skills: List[str]) -> List[Dict[str, Any]]:
    """Generate ATS (Applicant Tracking System) optimization suggestions."""
    
    suggestions = []
    
    # Check for common ATS issues
    if len(skills) < 5:
        suggestions.append({
            "type": "ats_skills",
            "title": "Add more technical skills",
            "description": f"You have {len(skills)} skills listed - ATS systems look for 8-12 skills",
            "priority": "medium",
            "action": "Add more relevant technical skills and tools you've used",
            "impact": "Medium - More skills increase keyword matching"
        })
    
    # Check for special characters that might confuse ATS
    problematic_chars = ["@", "#", "&", "%"]
    if any(char in resume_text for char in problematic_chars):
        suggestions.append({
            "type": "ats_format",
            "title": "Simplify special characters",
            "description": "Some special characters may confuse ATS systems",
            "priority": "low",
            "action": "Replace special characters with words (e.g., '&' with 'and')",
            "impact": "Low - Ensures better ATS compatibility"
        })
    
    return suggestions


def _calculate_overall_score(resume_text: str, skills: List[str]) -> Dict[str, Any]:
    """Calculate an overall resume score."""
    
    score = 0
    max_score = 100
    feedback = []
    
    # Content completeness (40 points)
    word_count = len(resume_text.split())
    if word_count >= 300:
        score += 20
        feedback.append("✅ Good content length")
    else:
        feedback.append("⚠️ Consider adding more detail")
    
    if "@" in resume_text:
        score += 10
        feedback.append("✅ Contact information present")
    else:
        feedback.append("❌ Missing contact information")
    
    if any(char.isdigit() for char in resume_text):
        score += 10
        feedback.append("✅ Contains quantifiable achievements")
    else:
        feedback.append("⚠️ Add numbers and metrics")
    
    # Skills and keywords (30 points)
    if len(skills) >= 8:
        score += 20
        feedback.append("✅ Good skill coverage")
    elif len(skills) >= 5:
        score += 15
        feedback.append("⚠️ Consider adding more skills")
    else:
        feedback.append("❌ Need more skills listed")
    
    # Check for action verbs (10 points)
    action_verbs = ["developed", "implemented", "created", "managed", "led", "optimized"]
    if any(verb in resume_text.lower() for verb in action_verbs):
        score += 10
        feedback.append("✅ Uses strong action verbs")
    else:
        feedback.append("⚠️ Use more action verbs")
    
    # Professional formatting (20 points)
    sections = ["experience", "education", "skills"]
    present_sections = sum(1 for section in sections if section.lower() in resume_text.lower())
    score += (present_sections / len(sections)) * 20
    
    if present_sections == len(sections):
        feedback.append("✅ Well-structured sections")
    else:
        feedback.append("⚠️ Add standard resume sections")
    
    return {
        "score": min(score, max_score),
        "max_score": max_score,
        "percentage": min(score / max_score * 100, 100),
        "feedback": feedback,
        "grade": _get_score_grade(score / max_score * 100)
    }


def _get_score_grade(percentage: float) -> str:
    """Get letter grade based on percentage."""
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"


async def _generate_priority_actions(
    evaluation_engine,
    resume_text: str,
    skills: List[str],
    target_role: Optional[str]
) -> List[Dict[str, Any]]:
    """Generate top 3 priority actions for immediate improvement."""
    
    actions = []
    
    # Priority 1: Contact information
    if "@" not in resume_text:
        actions.append({
            "priority": 1,
            "title": "Add contact information",
            "description": "Include email and phone number",
            "impact": "Critical",
            "time_estimate": "2 minutes"
        })
    
    # Priority 2: Quantifiable achievements
    if not any(char.isdigit() for char in resume_text):
        actions.append({
            "priority": 2,
            "title": "Add quantifiable achievements",
            "description": "Include numbers, percentages, and metrics",
            "impact": "High",
            "time_estimate": "15 minutes"
        })
    
    # Priority 3: Skills gap
    if len(skills) < 5:
        actions.append({
            "priority": 3,
            "title": "Expand technical skills section",
            "description": "Add more relevant tools and technologies",
            "impact": "High",
            "time_estimate": "10 minutes"
        })
    
    # Ensure we always return 3 actions
    while len(actions) < 3:
        actions.append({
            "priority": len(actions) + 1,
            "title": "Review and polish content",
            "description": "Check for typos and improve clarity",
            "impact": "Medium",
            "time_estimate": "10 minutes"
        })
    
    return actions[:3]