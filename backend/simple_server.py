"""
Simplified FastAPI application for testing
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import time
import hashlib
import secrets
from typing import Optional

# Create FastAPI application
app = FastAPI(
    title="Resume AI Server",
    version="1.0.0",
    description="AI-powered resume suggestions API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for requests
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class AISuggestionsRequest(BaseModel):
    resume_text: str
    target_role: Optional[str] = None

# Mock database for users
users_db = {
    "test@example.com": {
        "id": "1",
        "email": "test@example.com",
        "name": "Test User",
        "password_hash": "hashed_password_123",  # In real app, this would be properly hashed
        "created_at": "2024-01-01T00:00:00Z"
    }
}

# Mock tokens storage
active_tokens = {}

def hash_password(password: str) -> str:
    """Simple password hashing (use proper hashing in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token() -> str:
    """Generate a simple token"""
    return secrets.token_urlsafe(32)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Resume AI Server is running", "status": "success"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Server is running properly"
    }

# Authentication endpoints
@app.post("/api/v1/auth/login")
async def login(request: LoginRequest):
    """Login endpoint"""
    try:
        user = users_db.get(request.email)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Simple password check (use proper password verification in production)
        password_hash = hash_password(request.password)
        if user["password_hash"] != password_hash and request.password != "password123":
            # Allow "password123" for demo purposes
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Generate token
        token = generate_token()
        active_tokens[token] = user["id"]
        
        return {
            "status": "success",
            "data": {
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "id": user["id"],
                    "email": user["email"],
                    "name": user["name"]
                }
            },
            "message": "Login successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/api/v1/auth/register")
async def register(request: RegisterRequest):
    """Register endpoint"""
    try:
        # Check if user already exists
        if request.email in users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        user_id = str(len(users_db) + 1)
        password_hash = hash_password(request.password)
        
        users_db[request.email] = {
            "id": user_id,
            "email": request.email,
            "name": request.name,
            "password_hash": password_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        # Generate token
        token = generate_token()
        active_tokens[token] = user_id
        
        return {
            "status": "success",
            "data": {
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "id": user_id,
                    "email": request.email,
                    "name": request.name
                }
            },
            "message": "Registration successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.get("/api/v1/auth/me")
async def get_current_user():
    """Get current user info (simplified - normally would check authorization header)"""
    # For demo purposes, return a sample user
    return {
        "status": "success",
        "data": {
            "id": "1",
            "email": "test@example.com",
            "name": "Test User",
            "created_at": "2024-01-01T00:00:00Z"
        }
    }

@app.post("/api/v1/ai-suggestions")
async def ai_suggestions(request: AISuggestionsRequest):
    """AI suggestions endpoint"""
    
    # Simulate processing delay
    await asyncio.sleep(2)
    
    # Mock response
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

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8001)