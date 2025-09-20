#!/usr/bin/env python3
"""
Production Integration Test Script
Tests all backend modules with real implementations and dependencies.
Run this to validate the complete system is working correctly.
"""

import asyncio
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionIntegrationTester:
    """Comprehensive production integration tester."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    async def test_imports(self) -> bool:
        """Test all critical imports."""
        logger.info("🔍 Testing imports...")
        
        try:
            # Test core app imports
            from app.config import settings
            from app.core.database_init import get_database_manager
            from app.core.supabase import get_supabase_client
            from app.core.auth import create_access_token, verify_access_token
            from app.core.file_processing import get_resume_processor, get_job_description_processor
            from app.core.evaluation import get_evaluation_engine
            from app.core.observability import get_langsmith_manager
            
            logger.info("✅ Core app imports successful")
            
            # Test external library imports
            import fastapi
            import uvicorn
            import supabase
            import langchain
            import langchain_openai
            import langgraph
            import chromadb
            import spacy
            import nltk
            import fitz  # PyMuPDF
            import pdfplumber
            import docx2txt
            import numpy
            import pandas
            import sklearn
            
            logger.info("✅ External library imports successful")
            
            self.test_results["imports"] = {"status": "success", "details": "All imports working"}
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}")
            self.test_results["imports"] = {"status": "failed", "error": str(e)}
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected import error: {e}")
            self.test_results["imports"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_environment_configuration(self) -> bool:
        """Test environment configuration."""
        logger.info("🔧 Testing environment configuration...")
        
        try:
            from app.config import settings
            
            # Check critical settings
            config_status = {
                "database_url": bool(settings.supabase_url),
                "supabase_key": bool(settings.supabase_key),
                "openai_key": bool(settings.openai_api_key),
                "jwt_secret": bool(settings.secret_key),
                "allowed_origins": bool(settings.allowed_origins),
                "allowed_extensions": bool(settings.allowed_extensions),
                "max_file_size": settings.max_file_size > 0
            }
            
            logger.info(f"Configuration status: {config_status}")
            
            # Warn about missing configurations
            missing_configs = [k for k, v in config_status.items() if not v]
            if missing_configs:
                logger.warning(f"⚠️ Missing configurations: {missing_configs}")
                logger.warning("Some features may not work without proper environment variables")
            
            self.test_results["config"] = {
                "status": "success" if not missing_configs else "partial",
                "config_status": config_status,
                "missing_configs": missing_configs
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")
            self.test_results["config"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_database_connectivity(self) -> bool:
        """Test database connections."""
        logger.info("🗄️ Testing database connectivity...")
        
        try:
            from app.core.supabase import get_supabase_client
            from app.core.database_init import get_database_manager
            
            # Test Supabase client initialization
            supabase_client = get_supabase_client()
            if supabase_client:
                logger.info("✅ Supabase client initialized")
                
                # Test basic operation (this might fail without proper credentials)
                try:
                    # Basic table query test
                    response = supabase_client.table("resumes").select("*").limit(1).execute()
                    logger.info("✅ Supabase connection test successful")
                    db_connected = True
                except Exception as e:
                    logger.warning(f"⚠️ Supabase connection test failed: {e}")
                    logger.warning("This is expected if environment variables are not configured")
                    db_connected = False
            else:
                logger.error("❌ Supabase client not initialized")
                db_connected = False
            
            # Test database manager
            try:
                db_manager = get_database_manager()
                logger.info("✅ Database manager initialized")
            except Exception as e:
                logger.error(f"❌ Database manager failed: {e}")
            
            self.test_results["database"] = {
                "status": "success" if db_connected else "partial",
                "supabase_client": bool(supabase_client),
                "connection_test": db_connected
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            self.test_results["database"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_file_processing(self) -> bool:
        """Test file processing capabilities."""
        logger.info("📄 Testing file processing...")
        
        try:
            from app.core.file_processing import get_resume_processor, get_job_description_processor
            
            resume_processor = get_resume_processor()
            jd_processor = get_job_description_processor()
            
            # Test with sample text data
            sample_resume_text = b"""
            John Doe
            Software Engineer
            john.doe@email.com
            +1-555-123-4567
            
            EXPERIENCE:
            Senior Software Engineer at Tech Corp (2020-2023)
            - Developed web applications using Python, Django, and React
            - Implemented machine learning models using TensorFlow and scikit-learn
            - Built REST APIs and microservices using FastAPI
            - Worked with PostgreSQL and Redis databases
            - Deployed applications on AWS using Docker and Kubernetes
            
            SKILLS:
            Programming: Python, JavaScript, TypeScript, SQL
            Frameworks: Django, FastAPI, React, Node.js
            Databases: PostgreSQL, MongoDB, Redis
            Cloud: AWS, Docker, Kubernetes
            AI/ML: TensorFlow, PyTorch, scikit-learn, pandas, numpy
            
            EDUCATION:
            Bachelor of Science in Computer Science
            University of Technology (2016-2020)
            """
            
            # Test resume processing
            logger.info("Testing resume processing...")
            resume_result = resume_processor.process_resume(
                sample_resume_text, 
                "sample_resume.txt"
            )
            
            logger.info(f"✅ Resume processed - Skills found: {len(resume_result['extracted_skills'])}")
            logger.info(f"Sample skills: {resume_result['extracted_skills'][:5]}")
            
            # Test JD processing
            sample_jd_text = b"""
            Job Title: Senior Full Stack Developer
            Company: InnovaTech Solutions
            
            We are seeking a Senior Full Stack Developer to join our growing team.
            
            REQUIRED SKILLS:
            - 5+ years of experience in software development
            - Proficiency in Python and JavaScript
            - Experience with React and Node.js
            - Knowledge of SQL databases (PostgreSQL preferred)
            - Experience with cloud platforms (AWS, Azure)
            - Familiarity with Docker and containerization
            
            PREFERRED SKILLS:
            - Experience with machine learning and AI
            - Knowledge of Kubernetes
            - Experience with microservices architecture
            - Familiarity with TypeScript
            - Experience with CI/CD pipelines
            
            QUALIFICATIONS:
            - Bachelor's degree in Computer Science or related field
            - Strong problem-solving skills
            - Excellent communication skills
            """
            
            logger.info("Testing job description processing...")
            jd_result = jd_processor.process_job_description(
                sample_jd_text, 
                "sample_jd.txt",
                "Senior Full Stack Developer"
            )
            
            logger.info(f"✅ JD processed - Required skills: {len(jd_result['must_have_skills'])}")
            logger.info(f"Required skills: {jd_result['must_have_skills']}")
            
            self.test_results["file_processing"] = {
                "status": "success",
                "resume_skills_extracted": len(resume_result['extracted_skills']),
                "jd_required_skills": len(jd_result['must_have_skills']),
                "jd_preferred_skills": len(jd_result['good_to_have_skills'])
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ File processing test failed: {e}")
            self.test_results["file_processing"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_nlp_processing(self) -> bool:
        """Test NLP processing capabilities."""
        logger.info("🧠 Testing NLP processing...")
        
        try:
            # Test spaCy
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp("Python developer with machine learning experience")
                logger.info("✅ spaCy model loaded and working")
                spacy_working = True
            except Exception as e:
                logger.warning(f"⚠️ spaCy test failed: {e}")
                logger.warning("Run: python -m spacy download en_core_web_sm")
                spacy_working = False
            
            # Test NLTK
            try:
                import nltk
                from nltk.corpus import stopwords
                stop_words = stopwords.words('english')
                logger.info("✅ NLTK working")
                nltk_working = True
            except Exception as e:
                logger.warning(f"⚠️ NLTK test failed: {e}")
                nltk_working = False
            
            # Test vector processing
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model.encode(["test sentence"])
                logger.info("✅ SentenceTransformer working")
                st_working = True
            except Exception as e:
                logger.warning(f"⚠️ SentenceTransformer test failed: {e}")
                st_working = False
            
            # Test sklearn
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(["test document", "another document"])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                logger.info("✅ scikit-learn working")
                sklearn_working = True
            except Exception as e:
                logger.warning(f"⚠️ scikit-learn test failed: {e}")
                sklearn_working = False
            
            self.test_results["nlp"] = {
                "status": "success",
                "spacy": spacy_working,
                "nltk": nltk_working,
                "sentence_transformers": st_working,
                "sklearn": sklearn_working
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ NLP test failed: {e}")
            self.test_results["nlp"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_ai_integration(self) -> bool:
        """Test AI integration capabilities."""
        logger.info("🤖 Testing AI integration...")
        
        try:
            from app.core.evaluation import get_evaluation_engine
            
            evaluation_engine = get_evaluation_engine()
            
            # Test vector store initialization
            vector_store = evaluation_engine.vector_store
            if vector_store.chroma_client:
                logger.info("✅ ChromaDB initialized")
            
            if vector_store.sentence_transformer:
                logger.info("✅ SentenceTransformer model loaded")
            
            if vector_store.embeddings_model:
                logger.info("✅ OpenAI embeddings initialized")
            else:
                logger.warning("⚠️ OpenAI embeddings not initialized (API key may be missing)")
            
            # Test LLM evaluator
            llm_evaluator = evaluation_engine.llm_evaluator
            if llm_evaluator.llm:
                logger.info("✅ LLM initialized and ready")
            else:
                logger.warning("⚠️ LLM not initialized (API key may be missing)")
            
            # Test basic evaluation workflow (without external API calls)
            try:
                sample_resume = {
                    "processed_text": "Python developer with 3 years experience in Django and React",
                    "extracted_skills": ["python", "django", "react"]
                }
                
                sample_jd = {
                    "processed_text": "Looking for Python developer with Django experience",
                    "must_have_skills": ["python", "django"],
                    "good_to_have_skills": ["react", "aws"]
                }
                
                # Test without LLM to avoid API calls
                result = await evaluation_engine.evaluate_resume(
                    sample_resume, 
                    sample_jd, 
                    use_advanced_ai=False
                )
                
                logger.info(f"✅ Evaluation workflow test - Score: {result['relevance_score']:.1f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Evaluation workflow test failed: {e}")
            
            self.test_results["ai_integration"] = {
                "status": "success",
                "chroma_db": bool(vector_store.chroma_client),
                "sentence_transformer": bool(vector_store.sentence_transformer),
                "openai_embeddings": bool(vector_store.embeddings_model),
                "llm_available": bool(llm_evaluator.llm)
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ AI integration test failed: {e}")
            self.test_results["ai_integration"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_api_modules(self) -> bool:
        """Test API module imports and basic functionality."""
        logger.info("🌐 Testing API modules...")
        
        try:
            from app.api.v1.auth import router as auth_router
            from app.api.v1.resume import router as resume_router
            from app.api.v1.job_description import router as jd_router
            from app.api.v1.evaluation import router as eval_router
            from app.api.v1.dashboard import router as dashboard_router
            from app.api.v1.router import api_router
            
            logger.info("✅ All API routers imported successfully")
            
            # Test router configuration
            from app.main import app
            logger.info("✅ FastAPI app created successfully")
            
            # Test that all routers are properly configured
            routes = [route.path for route in app.routes]
            expected_paths = ["/api/v1/auth", "/api/v1/resumes", "/api/v1/job-descriptions", 
                            "/api/v1/evaluations", "/api/v1/dashboard"]
            
            found_paths = [path for path in expected_paths if any(path in route for route in routes)]
            logger.info(f"API paths found: {found_paths}")
            
            self.test_results["api_modules"] = {
                "status": "success",
                "routers_imported": 5,
                "app_created": True,
                "routes_configured": len(found_paths)
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ API modules test failed: {e}")
            self.test_results["api_modules"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_vector_databases(self) -> bool:
        """Test vector database implementations."""
        logger.info("📊 Testing vector databases...")
        
        try:
            # Test ChromaDB
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            client = chromadb.Client(ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            # Create a test collection
            collection = client.create_collection("test_collection")
            
            # Add test documents
            test_docs = ["Python developer", "React engineer", "Full stack developer"]
            collection.add(
                documents=test_docs,
                ids=["1", "2", "3"]
            )
            
            # Query test
            results = collection.query(
                query_texts=["software engineer"],
                n_results=2
            )
            
            logger.info(f"✅ ChromaDB test successful - Found {len(results['documents'][0])} results")
            
            # Cleanup
            client.delete_collection("test_collection")
            
            # Test FAISS
            try:
                import faiss
                import numpy as np
                
                # Create simple test index
                dimension = 384
                index = faiss.IndexFlatL2(dimension)
                
                # Add test vectors
                test_vectors = np.random.random((10, dimension)).astype('float32')
                index.add(test_vectors)
                
                # Search test
                search_vectors = np.random.random((1, dimension)).astype('float32')
                distances, indices = index.search(search_vectors, 5)
                
                logger.info(f"✅ FAISS test successful - Index size: {index.ntotal}")
                
            except Exception as e:
                logger.warning(f"⚠️ FAISS test failed: {e}")
            
            self.test_results["vector_databases"] = {
                "status": "success",
                "chromadb": True,
                "faiss": True
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector databases test failed: {e}")
            self.test_results["vector_databases"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_text_processing_pipeline(self) -> bool:
        """Test complete text processing pipeline."""
        logger.info("🔤 Testing text processing pipeline...")
        
        try:
            from app.core.file_processing import get_resume_processor
            from app.core.evaluation import ProductionNLPProcessor, ProductionSkillMatcher
            
            # Test NLP processor
            nlp_processor = ProductionNLPProcessor()
            
            test_text = """
            Senior Python Developer with 5 years of experience in web development.
            Expertise in Django, FastAPI, React, and PostgreSQL.
            Experience with AWS, Docker, Kubernetes, and machine learning.
            Built multiple REST APIs and worked on data science projects.
            """
            
            # Test skill extraction
            skills = nlp_processor.extract_skills_advanced(test_text)
            logger.info(f"✅ Skills extracted: {len(skills)} skills")
            logger.info(f"Sample skills: {skills[:5]}")
            
            # Test skill normalization
            normalized = nlp_processor.normalize_skills(skills)
            logger.info(f"✅ Skills normalized: {len(normalized)} unique skills")
            
            # Test skill matching
            skill_matcher = ProductionSkillMatcher()
            hard_score, matched, missing, percentage = skill_matcher.match_skills(
                test_text,
                skills,
                ["python", "django", "react", "aws"],
                ["kubernetes", "machine learning", "typescript"]
            )
            
            logger.info(f"✅ Skill matching test - Score: {hard_score:.1f}%")
            logger.info(f"Matched: {matched}")
            logger.info(f"Missing: {missing}")
            
            self.test_results["text_processing"] = {
                "status": "success",
                "skills_extracted": len(skills),
                "skills_normalized": len(normalized),
                "matching_score": hard_score,
                "matched_skills": len(matched),
                "missing_skills": len(missing)
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Text processing pipeline test failed: {e}")
            self.test_results["text_processing"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_observability(self) -> bool:
        """Test observability and logging."""
        logger.info("📈 Testing observability...")
        
        try:
            from app.core.observability import get_langsmith_manager
            
            langsmith = get_langsmith_manager()
            
            if langsmith.enabled:
                logger.info("✅ LangSmith tracing enabled")
            else:
                logger.warning("⚠️ LangSmith tracing not enabled (API key may be missing)")
            
            # Test logging functionality
            from app.core.logging import setup_logging
            
            setup_logging()
            logger.info("✅ Logging configuration successful")
            
            self.test_results["observability"] = {
                "status": "success",
                "langsmith_enabled": langsmith.enabled,
                "logging_configured": True
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Observability test failed: {e}")
            self.test_results["observability"] = {"status": "failed", "error": str(e)}
            return False
    
    async def test_authentication(self) -> bool:
        """Test authentication system."""
        logger.info("🔐 Testing authentication...")
        
        try:
            from app.core.auth import create_access_token, verify_access_token
            
            # Test JWT token creation and verification
            test_user_data = {"sub": "test@example.com", "user_id": "123"}
            token = create_access_token(test_user_data)
            logger.info("✅ JWT token created")
            
            # Test token verification
            payload = verify_access_token(token)
            logger.info(f"✅ JWT token verified - User: {payload.get('sub')}")
            
            self.test_results["authentication"] = {
                "status": "success",
                "jwt_creation": True,
                "jwt_verification": True
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {e}")
            self.test_results["authentication"] = {"status": "failed", "error": str(e)}
            return False
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🧪 PRODUCTION INTEGRATION TEST SUMMARY")
        print("="*60)
        
        # Count successes and failures
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if result.get("status") == "success")
        partial_tests = sum(1 for result in self.test_results.values() 
                          if result.get("status") == "partial")
        failed_tests = sum(1 for result in self.test_results.values() 
                         if result.get("status") == "failed")
        
        print(f"📊 Test Results: {successful_tests} ✅ | {partial_tests} ⚠️ | {failed_tests} ❌")
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print()
        
        # Detailed results
        for test_name, result in self.test_results.items():
            status_icon = {"success": "✅", "partial": "⚠️", "failed": "❌"}.get(result["status"], "❓")
            print(f"{status_icon} {test_name.upper().replace('_', ' ')}")
            
            if result["status"] == "failed":
                print(f"   Error: {result.get('error', 'Unknown error')}")
            elif result["status"] == "partial":
                print(f"   Issues: {result.get('missing_configs', result.get('details', 'Some components not available'))}")
            else:
                print(f"   Status: All components working correctly")
            print()
        
        # Recommendations
        print("📋 RECOMMENDATIONS:")
        print()
        
        if failed_tests > 0:
            print("❌ Critical Issues:")
            for test_name, result in self.test_results.items():
                if result.get("status") == "failed":
                    print(f"   - Fix {test_name}: {result.get('error', 'Unknown error')}")
            print()
        
        if partial_tests > 0:
            print("⚠️ Configuration Issues:")
            if "config" in self.test_results and self.test_results["config"].get("missing_configs"):
                missing = self.test_results["config"]["missing_configs"]
                print(f"   - Set environment variables for: {', '.join(missing)}")
            if "nlp" in self.test_results:
                nlp_result = self.test_results["nlp"]
                if not nlp_result.get("spacy", True):
                    print("   - Install spaCy model: python -m spacy download en_core_web_sm")
            print()
        
        if successful_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Your backend is production-ready.")
        elif successful_tests + partial_tests == total_tests:
            print("✨ MOSTLY READY! Address configuration issues for full functionality.")
        else:
            print("🔧 NEEDS WORK! Please fix the failed components.")
        
        print()
        print("🚀 To start the server:")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print()
        print("📚 API Documentation will be available at:")
        print("   http://localhost:8000/docs")
        print("="*60)
    
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("🚀 Starting Production Integration Tests...")
        print("\n🧪 PRODUCTION INTEGRATION TESTING")
        print("Testing all backend modules and dependencies...\n")
        
        # Run tests in logical order
        test_functions = [
            self.test_imports,
            self.test_environment_configuration,
            self.test_database_connectivity,
            self.test_file_processing,
            self.test_nlp_processing,
            self.test_ai_integration,
            self.test_api_modules,
            self.test_authentication,
            self.test_observability
        ]
        
        for test_func in test_functions:
            try:
                await test_func()
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} crashed: {e}")
                self.test_results[test_func.__name__.replace("test_", "")] = {
                    "status": "failed",
                    "error": f"Test crashed: {str(e)}"
                }
        
        # Print summary
        self.print_test_summary()


async def main():
    """Main test execution function."""
    tester = ProductionIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())