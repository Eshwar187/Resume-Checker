from fastapi import APIRouter

from app.api.v1 import auth, resume, job_description, evaluation, dashboard, ai_suggestions


api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(resume.router)
api_router.include_router(job_description.router)
api_router.include_router(evaluation.router)
api_router.include_router(dashboard.router)
api_router.include_router(ai_suggestions.router)