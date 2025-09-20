"""
Main API router for v1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1 import auth, resume, job_description, evaluation, dashboard

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router)
api_router.include_router(resume.router)
api_router.include_router(job_description.router)
api_router.include_router(evaluation.router)
api_router.include_router(dashboard.router)