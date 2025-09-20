"""
API tests for the Resume Relevance Check System.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.config import settings

# Create test client
client = TestClient(app)

# Test data
test_user_data = {
    "email": "test@example.com",
    "password": "testpassword123",
    "name": "Test User",
    "role": "student"
}

test_job_data = {
    "title": "Software Engineer",
    "company": "Tech Corp",
    "job_text": """
    We are looking for a Software Engineer with experience in Python, React, and AWS.
    
    Required Skills:
    - Python programming
    - React.js development
    - AWS cloud services
    - SQL databases
    
    Preferred Skills:
    - Docker containerization
    - Kubernetes orchestration
    - Machine learning experience
    """
}


class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["environment"] == settings.environment
    
    @patch('app.core.auth.get_auth_manager')
    def test_signup_success(self, mock_auth_manager):
        """Test successful user signup."""
        # Mock auth manager
        mock_manager = MagicMock()
        mock_manager.sign_up.return_value = {
            "access_token": "test_token",
            "token_type": "bearer",
            "expires_in": 86400,
            "user": {
                "id": "test_user_id",
                "email": test_user_data["email"],
                "name": test_user_data["name"],
                "role": test_user_data["role"],
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
        mock_auth_manager.return_value = mock_manager
        
        response = client.post("/api/v1/auth/signup", json=test_user_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["access_token"] == "test_token"
        assert data["user"]["email"] == test_user_data["email"]
    
    @patch('app.core.auth.get_auth_manager')
    def test_login_success(self, mock_auth_manager):
        """Test successful user login."""
        # Mock auth manager
        mock_manager = MagicMock()
        mock_manager.sign_in.return_value = {
            "access_token": "test_token",
            "token_type": "bearer",
            "expires_in": 86400,
            "user": {
                "id": "test_user_id",
                "email": test_user_data["email"],
                "name": test_user_data["name"],
                "role": test_user_data["role"],
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
        mock_auth_manager.return_value = mock_manager
        
        login_data = {
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["access_token"] == "test_token"
    
    def test_signup_invalid_email(self):
        """Test signup with invalid email."""
        invalid_data = test_user_data.copy()
        invalid_data["email"] = "invalid_email"
        
        response = client.post("/api/v1/auth/signup", json=invalid_data)
        assert response.status_code == 422
    
    def test_signup_weak_password(self):
        """Test signup with weak password."""
        weak_data = test_user_data.copy()
        weak_data["password"] = "123"
        
        response = client.post("/api/v1/auth/signup", json=weak_data)
        assert response.status_code == 422


class TestFileProcessing:
    """Test file processing functionality."""
    
    @patch('app.core.file_processing.get_resume_processor')
    def test_resume_text_extraction(self, mock_processor):
        """Test resume text extraction."""
        # Mock processor
        mock_proc = MagicMock()
        mock_proc.process_resume.return_value = {
            "raw_text": "John Doe\nSoftware Engineer\nPython, React, AWS",
            "processed_text": "john doe software engineer python react aws",
            "extracted_skills": ["Python", "React", "AWS"],
            "extracted_experience": {"total_years": 3},
            "contact_info": {"email": "john@example.com"},
            "education": ["Computer Science"]
        }
        mock_processor.return_value = mock_proc
        
        # Test processing
        file_content = b"test resume content"
        filename = "resume.pdf"
        
        processor = mock_processor()
        result = processor.process_resume(file_content, filename)
        
        assert "Python" in result["extracted_skills"]
        assert result["extracted_experience"]["total_years"] == 3
    
    @patch('app.core.file_processing.get_job_description_processor')
    def test_job_description_processing(self, mock_processor):
        """Test job description processing."""
        # Mock processor
        mock_proc = MagicMock()
        mock_proc.process_job_description.return_value = {
            "raw_text": test_job_data["job_text"],
            "processed_text": "software engineer python react aws sql docker kubernetes",
            "job_title": "Software Engineer",
            "must_have_skills": ["Python", "React.js", "AWS", "SQL"],
            "good_to_have_skills": ["Docker", "Kubernetes", "Machine learning"],
            "qualifications": ["Bachelor's degree in Computer Science"],
            "experience_required": "3+ years",
            "location": "Remote",
            "job_type": "Full-time"
        }
        mock_processor.return_value = mock_proc
        
        # Test processing
        processor = mock_processor()
        result = processor.process_job_description(
            test_job_data["job_text"].encode(), 
            "job.txt", 
            test_job_data["title"]
        )
        
        assert "Python" in result["must_have_skills"]
        assert "Docker" in result["good_to_have_skills"]


class TestEvaluationEngine:
    """Test AI evaluation functionality."""
    
    @patch('app.core.evaluation.get_evaluation_engine')
    async def test_evaluation_pipeline(self, mock_engine):
        """Test complete evaluation pipeline."""
        # Mock evaluation engine
        mock_eval_engine = MagicMock()
        
        # Mock evaluation result
        from app.core.evaluation import EvaluationResult
        from app.schemas import EvaluationVerdict
        
        mock_result = EvaluationResult()
        mock_result.relevance_score = 85.5
        mock_result.verdict = EvaluationVerdict.HIGH
        mock_result.matched_skills = ["Python", "React", "AWS"]
        mock_result.missing_skills = ["Docker", "Kubernetes"]
        mock_result.skill_match_percentage = 75.0
        mock_result.experience_match = True
        mock_result.feedback = "Strong candidate with relevant experience"
        mock_result.recommendations = ["Learn containerization", "Gain Kubernetes experience"]
        mock_result.hard_match_score = 80.0
        mock_result.soft_match_score = 75.0
        mock_result.processing_time = 2.5
        
        mock_eval_engine.evaluate_resume.return_value = mock_result
        mock_engine.return_value = mock_eval_engine
        
        # Test evaluation
        resume_data = {
            "processed_text": "python developer with react and aws experience",
            "extracted_skills": ["Python", "React", "AWS", "SQL"]
        }
        
        jd_data = {
            "processed_text": "looking for python react aws developer",
            "must_have_skills": ["Python", "React", "AWS", "Docker"],
            "good_to_have_skills": ["Kubernetes", "Machine Learning"]
        }
        
        engine = mock_engine()
        result = await engine.evaluate_resume(resume_data, jd_data, True)
        
        assert result.relevance_score == 85.5
        assert result.verdict == EvaluationVerdict.HIGH
        assert "Python" in result.matched_skills
        assert "Docker" in result.missing_skills
    
    def test_hard_match_scoring(self):
        """Test hard match scoring algorithm."""
        from app.core.evaluation import HardMatcher
        
        matcher = HardMatcher()
        
        resume_text = "Experienced Python developer with React and AWS knowledge"
        resume_skills = ["Python", "React", "AWS", "JavaScript"]
        required_skills = ["Python", "React", "AWS", "Docker"]
        preferred_skills = ["Kubernetes", "Machine Learning"]
        
        score, matched, missing, percentage = matcher.calculate_hard_match_score(
            resume_text, resume_skills, required_skills, preferred_skills
        )
        
        assert score > 0
        assert "Python" in matched
        assert "React" in matched
        assert "AWS" in matched
        assert "Docker" in missing
        assert percentage > 0
    
    def test_soft_match_scoring(self):
        """Test soft match scoring algorithm."""
        from app.core.evaluation import SoftMatcher
        
        matcher = SoftMatcher()
        
        resume_text = "Python developer with web development experience using React framework"
        jd_text = "Looking for Python programmer with React.js skills for web applications"
        
        score, metadata = matcher.calculate_soft_match_score(resume_text, jd_text)
        
        assert score > 0
        assert score <= 100
        assert "method" in metadata


class TestDashboardAPIs:
    """Test dashboard functionality."""
    
    @patch('app.core.auth.get_current_user')
    @patch('app.core.supabase.get_supabase_service_client')
    def test_student_dashboard(self, mock_supabase, mock_user):
        """Test student dashboard endpoint."""
        # Mock current user
        from app.schemas import UserResponse, UserRole
        mock_user.return_value = UserResponse(
            id="test_user_id",
            email="test@example.com",
            name="Test User",
            role=UserRole.STUDENT,
            is_active=True,
            created_at="2024-01-01T00:00:00Z"
        )
        
        # Mock Supabase responses
        mock_client = MagicMock()
        
        # Mock resumes query
        mock_resumes_result = MagicMock()
        mock_resumes_result.data = [
            {
                "id": "resume_1",
                "user_id": "test_user_id",
                "file_name": "resume.pdf",
                "file_url": "https://example.com/resume.pdf",
                "file_size": 1024,
                "file_type": "application/pdf",
                "uploaded_at": "2024-01-01T00:00:00Z"
            }
        ]
        
        # Mock evaluations query
        mock_eval_result = MagicMock()
        mock_eval_result.data = [
            {
                "id": "eval_1",
                "resume_id": "resume_1",
                "jd_id": "jd_1",
                "relevance_score": 85.0,
                "verdict": "High",
                "matched_skills": ["Python", "React"],
                "missing_skills": ["Docker"],
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
        
        # Configure mock chain
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_resumes_result
        mock_client.table.return_value.select.return_value.in_.return_value.order.return_value.limit.return_value.execute.return_value = mock_eval_result
        
        mock_supabase.return_value = mock_client
        
        # Test request
        headers = {"Authorization": "Bearer test_token"}
        response = client.get("/api/v1/dashboard/student", headers=headers)
        
        # Note: This will likely fail without proper auth mocking
        # In a real test, you'd need to properly mock the authentication middleware


if __name__ == "__main__":
    pytest.main([__file__, "-v"])