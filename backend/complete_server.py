from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
import asyncio
import json
import time
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import jwt
import hashlib

# Create FastAPI application
app = FastAPI(
    title="Resume AI Server",
    version="1.0.0",
    description="Complete AI-powered resume analysis with authentication"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = "YGY/vCWEzyykTf/D7vMaasfAE/GdO9NeDiLvFtII/QAWbnjwUHTYEY3vGt8QcwSjbcRmOckkqk+FGpD411JKFQ=="
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3600

# Mock database
users_db = {}
resumes_db = {}
job_descriptions_db = {}
evaluations_db = {}

# Pydantic models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AIAnalysisRequest(BaseModel):
    resume_text: str
    target_role: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    id: str
    email: str
    name: str
    created_at: str

# Helper functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        if user_id not in users_db:
            raise HTTPException(status_code=401, detail="User not found")
            
        return users_db[user_id]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Root and health endpoints
@app.get("/")
async def root():
    return {"message": "Resume AI Server is running", "status": "success"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Server is running properly",
        "endpoints": {
            "auth": ["/api/v1/auth/register", "/api/v1/auth/login"],
            "upload": ["/api/v1/resumes/upload", "/api/v1/job-descriptions/upload"],
            "analysis": ["/api/v1/ai-suggestions"]
        }
    }

# Authentication endpoints
@app.post("/api/v1/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    # Check if user already exists
    for user in users_db.values():
        if user["email"] == user_data.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user_id = str(uuid.uuid4())
    users_db[user_id] = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password_hash": hash_password(user_data.password),
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Create access token
    access_token = create_access_token(data={"sub": user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/api/v1/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    # Find user by email
    user = None
    for u in users_db.values():
        if u["email"] == user_data.email:
            user = u
            break
    
    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user["id"]})
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/api/v1/auth/me", response_model=User)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return User(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        created_at=current_user["created_at"]
    )

# File upload endpoints
@app.post("/api/v1/resumes/upload")
async def upload_resume(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    # Read file content
    content = await file.read()
    
    # Mock text extraction (in real app, use PyMuPDF/pdfplumber)
    extracted_text = f"Sample extracted text from {file.filename}"
    
    # Save resume data
    resume_id = str(uuid.uuid4())
    resumes_db[resume_id] = {
        "id": resume_id,
        "user_id": current_user["id"],
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": len(content),
        "extracted_text": extracted_text,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    return {
        "status": "success",
        "data": {
            "id": resume_id,
            "filename": file.filename,
            "extracted_text": extracted_text,
            "file_size": len(content),
            "uploaded_at": resumes_db[resume_id]["uploaded_at"]
        }
    }

@app.get("/api/v1/resumes")
async def get_user_resumes(current_user: dict = Depends(get_current_user)):
    user_resumes = [
        resume for resume in resumes_db.values() 
        if resume["user_id"] == current_user["id"]
    ]
    return {"status": "success", "data": user_resumes}

@app.post("/api/v1/job-descriptions/upload")
async def upload_job_description(
    file: UploadFile = File(...),
    title: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported")
    
    content = await file.read()
    extracted_text = f"Sample JD text from {file.filename}"
    
    jd_id = str(uuid.uuid4())
    jd_data = {
        "id": jd_id,
        "user_id": current_user["id"],
        "title": title,
        "filename": file.filename,
        "extracted_text": extracted_text,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    # Store in database
    job_descriptions_db[jd_id] = jd_data
    
    return {"status": "success", "data": jd_data}

@app.get("/api/v1/job-descriptions")
async def get_user_job_descriptions(current_user: dict = Depends(get_current_user)):
    user_jds = [
        jd for jd in job_descriptions_db.values() 
        if jd["user_id"] == current_user["id"]
    ]
    return {"status": "success", "data": user_jds}

# AI Suggestions endpoint
@app.post("/api/v1/ai-suggestions")
async def generate_ai_suggestions(
    request: AIAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    # Simulate AI processing delay
    await asyncio.sleep(2)
    
    # Mock AI analysis response
    response = {
        "status": "success",
        "message": "AI suggestions generated successfully",
        "data": {
            "suggestions": {
                "overall_score": {
                    "score": 78,
                    "max_score": 100,
                    "percentage": 78.0,
                    "grade": "B+",
                    "feedback": [
                        "Strong technical skills demonstrated",
                        "Good project experience shown",
                        "Consider adding more quantified achievements"
                    ]
                },
                "priority_actions": [
                    {
                        "priority": 1,
                        "title": "Add Quantified Achievements",
                        "description": "Include specific numbers and metrics in your accomplishments",
                        "impact": "High",
                        "time_estimate": "2-3 hours"
                    },
                    {
                        "priority": 2,
                        "title": "Update Technical Skills",
                        "description": "Add trending technologies like Docker and Kubernetes",
                        "impact": "Medium",
                        "time_estimate": "1 hour"
                    }
                ],
                "skill_recommendations": [
                    {
                        "type": "skill_recommendations",
                        "title": "Add Docker & Containerization",
                        "description": "Docker is highly sought after for modern software development roles",
                        "action": "Include Docker, Kubernetes, and container orchestration in your skills section",
                        "priority": "high",
                        "impact": "Increases job match by 25%"
                    },
                    {
                        "type": "skill_recommendations",
                        "title": "Cloud Platforms",
                        "description": "Cloud experience is essential for modern development",
                        "action": "Add AWS, Azure, or Google Cloud Platform experience",
                        "priority": "medium",
                        "impact": "Expands job opportunities"
                    }
                ],
                "content_improvements": [
                    {
                        "type": "content_improvements",
                        "title": "Quantify Your Impact",
                        "description": "Add specific metrics to demonstrate your achievements",
                        "action": "Replace vague statements with quantified results (e.g., 'Improved performance by 40%')",
                        "priority": "high",
                        "impact": "Makes achievements more compelling"
                    },
                    {
                        "type": "content_improvements",
                        "title": "Add Leadership Examples",
                        "description": "Include examples of leadership and collaboration",
                        "action": "Describe team leadership, mentoring, or cross-functional collaboration experiences",
                        "priority": "medium",
                        "impact": "Shows soft skills and growth potential"
                    }
                ],
                "formatting_suggestions": [
                    {
                        "type": "formatting_suggestions",
                        "title": "Use Action Verbs",
                        "description": "Start bullet points with strong action verbs",
                        "action": "Begin each accomplishment with verbs like 'Developed', 'Implemented', 'Led', 'Optimized'",
                        "priority": "low",
                        "impact": "Makes content more engaging"
                    }
                ],
                "keyword_optimization": [
                    {
                        "type": "keyword_optimization",
                        "title": "Add Industry Keywords",
                        "description": "Include trending software engineering keywords",
                        "action": "Add terms like 'microservices', 'API development', 'agile methodologies'",
                        "priority": "medium",
                        "impact": "Improves ATS compatibility"
                    }
                ],
                "ats_optimization": [
                    {
                        "type": "ats_optimization",
                        "title": "Use Standard Section Headers",
                        "description": "Ensure ATS can parse your resume sections",
                        "action": "Use standard headers like 'EXPERIENCE', 'EDUCATION', 'SKILLS'",
                        "priority": "low",
                        "impact": "Improves ATS parsing accuracy"
                    }
                ]
            }
        }
    }
    
    return response

# Real-time AI Suggestions streaming (SSE)
@app.post("/api/v1/ai-suggestions/stream")
async def stream_ai_suggestions(
    request: AIAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    async def event_generator():
        try:
            # Step 1: Acknowledge and basic parsing
            yield f"data: {json.dumps({"event": "start", "message": "Starting analysis", "target_role": request.target_role})}\n\n"
            await asyncio.sleep(0.3)

            # Step 2: Skill recommendations
            skills = [
                {"type": "skill", "title": "Python", "priority": "high", "action": "Add concrete Python project bullets", "impact": "Higher ATS match"},
                {"type": "skill", "title": "FastAPI", "priority": "medium", "action": "Mention APIs built with metrics & auth", "impact": "Improves backend relevance"},
            ]
            yield f"data: {json.dumps({"event": "skill_recommendations", "items": skills})}\n\n"
            await asyncio.sleep(0.4)

            # Step 3: Content improvements
            content = [
                {"type": "content", "title": "Quantify impact", "description": "Use numbers (% latency drop, throughput, users)", "priority": "high"},
                {"type": "content", "title": "Action verbs", "description": "Start bullets with led, built, designed", "priority": "low"},
            ]
            yield f"data: {json.dumps({"event": "content_improvements", "items": content})}\n\n"
            await asyncio.sleep(0.4)

            # Step 4: Formatting/ATS tips
            formatting = [
                {"type": "format", "title": "Consistent headers", "description": "Keep H2 size uniform", "priority": "low"},
                {"type": "format", "title": "ATS-safe", "description": "Avoid text boxes; use simple bullets", "priority": "medium"},
            ]
            yield f"data: {json.dumps({"event": "formatting_suggestions", "items": formatting})}\n\n"
            await asyncio.sleep(0.4)

            # Step 5: Overall score (mocked)
            overall = {
                "score": 78,
                "max_score": 100,
                "percentage": 78,
                "grade": "B+",
                "feedback": [
                    "Good base; emphasize measurable outcomes",
                    "Add role-specific keywords for better match"
                ],
            }
            yield f"data: {json.dumps({"event": "overall_score", "value": overall})}\n\n"
            await asyncio.sleep(0.3)

            # Step 6: Priority actions
            actions = [
                {"priority": 1, "title": "Add 3 quantified bullets", "impact": "High", "time_estimate": "15m"},
                {"priority": 2, "title": "Include FastAPI + Docker keywords", "impact": "Medium", "time_estimate": "10m"},
            ]
            yield f"data: {json.dumps({"event": "priority_actions", "items": actions})}\n\n"
            await asyncio.sleep(0.2)

            # Done
            yield f"data: {json.dumps({"event": "done"})}\n\n"
        except Exception as e:
            err = {"event": "error", "message": str(e)}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# AI Suggestions endpoint without authentication (for testing)
@app.post("/api/v1/ai-suggestions-public")
async def generate_ai_suggestions_public(request: AIAnalysisRequest):
    # Same response as authenticated endpoint but without auth requirement
    await asyncio.sleep(1)  # Shorter delay for testing
    
    response = {
        "status": "success",
        "message": "AI suggestions generated successfully (public endpoint)",
        "data": {
            "suggestions": {
                "overall_score": {
                    "score": 82,
                    "max_score": 100,
                    "percentage": 82.0,
                    "grade": "B+",
                    "feedback": [
                        "✅ Strong technical foundation",
                        "✅ Good educational background", 
                        "⚠️ Add more quantified achievements",
                        "⚠️ Include more trending technologies"
                    ]
                },
                "priority_actions": [
                    {
                        "priority": 1,
                        "title": "Add Quantified Achievements",
                        "description": "Include specific numbers and metrics to show impact",
                        "impact": "High",
                        "time_estimate": "30 minutes"
                    },
                    {
                        "priority": 2,
                        "title": "Update Technical Skills",
                        "description": "Add Docker, Kubernetes, and cloud technologies",
                        "impact": "High",
                        "time_estimate": "15 minutes"
                    },
                    {
                        "priority": 3,
                        "title": "Enhance Project Descriptions",
                        "description": "Add more details about your projects and their impact",
                        "impact": "Medium",
                        "time_estimate": "45 minutes"
                    }
                ],
                "skill_recommendations": [
                    {
                        "type": "skill_recommendations",
                        "title": "Add Docker & Containerization",
                        "description": "Containerization skills are essential for modern development",
                        "action": "Learn Docker, Docker Compose, and Kubernetes basics",
                        "priority": "high",
                        "impact": "Increases job match by 30%"
                    },
                    {
                        "type": "skill_recommendations",
                        "title": "Cloud Platforms (AWS/Azure)",
                        "description": "Cloud experience is highly valued in today's market",
                        "action": "Get AWS or Azure certification, deploy projects to cloud",
                        "priority": "high",
                        "impact": "Opens 40% more job opportunities"
                    },
                    {
                        "type": "skill_recommendations", 
                        "title": "TypeScript",
                        "description": "TypeScript is becoming the standard for enterprise JavaScript",
                        "action": "Convert existing JavaScript projects to TypeScript",
                        "priority": "medium",
                        "impact": "Makes you eligible for senior roles"
                    }
                ],
                "content_improvements": [
                    {
                        "type": "content_improvements",
                        "title": "Quantify Your Achievements",
                        "description": "Numbers make your impact clear and memorable",
                        "action": "Add metrics like 'Improved performance by 40%', 'Reduced loading time by 2s'",
                        "priority": "high",
                        "impact": "Makes achievements 60% more compelling"
                    },
                    {
                        "type": "content_improvements",
                        "title": "Add Project Impact",
                        "description": "Show the business value of your technical work",
                        "action": "Describe user adoption, cost savings, or efficiency gains",
                        "priority": "medium",
                        "impact": "Demonstrates business awareness"
                    },
                    {
                        "type": "content_improvements",
                        "title": "Include Team Collaboration",
                        "description": "Show your ability to work in teams",
                        "action": "Mention cross-functional work, code reviews, mentoring",
                        "priority": "medium",
                        "impact": "Shows soft skills valued by employers"
                    }
                ],
                "formatting_suggestions": [
                    {
                        "type": "formatting_suggestions",
                        "title": "Use Consistent Bullet Points",
                        "description": "Consistent formatting makes your resume easier to scan",
                        "action": "Use the same bullet style throughout your resume",
                        "priority": "low",
                        "impact": "Improves readability"
                    },
                    {
                        "type": "formatting_suggestions",
                        "title": "Optimize Section Order",
                        "description": "Put your strongest sections first",
                        "action": "Order: Contact → Skills → Experience → Projects → Education",
                        "priority": "low", 
                        "impact": "Highlights your strengths early"
                    }
                ],
                "keyword_optimization": [
                    {
                        "type": "keyword_optimization",
                        "title": "Add Technical Keywords",
                        "description": "Include buzzwords that ATS systems look for",
                        "action": "Add 'microservices', 'RESTful APIs', 'agile', 'CI/CD'",
                        "priority": "medium",
                        "impact": "Improves ATS matching by 25%"
                    },
                    {
                        "type": "keyword_optimization",
                        "title": "Industry-Specific Terms",
                        "description": "Use terminology specific to software engineering",
                        "action": "Include 'full-stack', 'scalable', 'responsive design', 'database optimization'",
                        "priority": "medium",
                        "impact": "Shows domain expertise"
                    }
                ],
                "ats_optimization": [
                    {
                        "type": "ats_optimization",
                        "title": "Use Standard Section Headers",
                        "description": "ATS systems look for standard section names",
                        "action": "Use 'PROFESSIONAL EXPERIENCE', 'TECHNICAL SKILLS', 'EDUCATION'",
                        "priority": "medium",
                        "impact": "Improves ATS parsing accuracy"
                    },
                    {
                        "type": "ats_optimization",
                        "title": "Avoid Complex Formatting",
                        "description": "Keep formatting simple for ATS compatibility",
                        "action": "Use standard fonts, avoid tables/columns, keep text selectable",
                        "priority": "low",
                        "impact": "Ensures ATS can read your resume"
                    }
                ]
            }
        }
    }
    
    return response

# Dashboard endpoint
@app.get("/api/v1/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    user_resumes = [r for r in resumes_db.values() if r["user_id"] == current_user["id"]]
    
    return {
        "status": "success",
        "data": {
            "total_resumes": len(user_resumes),
            "total_evaluations": 0,
            "average_score": 78.5,
            "recent_activity": []
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Complete Resume AI Server...")
    print("📝 Available endpoints:")
    print("   POST /api/v1/auth/register - User registration")
    print("   POST /api/v1/auth/login - User login")
    print("   GET /api/v1/auth/me - Get current user")
    print("   POST /api/v1/resumes/upload - Upload resume")
    print("   GET /api/v1/resumes - Get user resumes")
    print("   POST /api/v1/job-descriptions/upload - Upload job description")
    print("   GET /api/v1/job-descriptions - Get user job descriptions")
    print("   POST /api/v1/ai-suggestions - AI analysis (authenticated)")
    print("   POST /api/v1/ai-suggestions/stream - AI analysis (SSE stream)")
    print("   POST /api/v1/ai-suggestions-public - AI analysis (public)")
    print("   GET /health - Health check")
    print("💡 Frontend should connect to: http://localhost:8003")
    
    uvicorn.run(app, host="0.0.0.0", port=8003)