"""
Database initialization and migration utilities.
"""

import asyncio
import logging
from typing import Dict, Any

from app.core.supabase import get_supabase_service_client
from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database initialization and migrations."""
    
    def __init__(self):
        self.client = get_supabase_service_client()
    
    async def initialize_database(self):
        """Initialize database with required tables and data."""
        try:
            await self.create_tables()
            await self.create_storage_buckets()
            await self.setup_row_level_security()
            await self.create_initial_data()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables if they don't exist."""
        # This would typically be done via Supabase SQL editor or migrations
        # For now, we'll log the required SQL
        
        sql_statements = [
            """
            -- Users table (extends Supabase auth.users)
            CREATE TABLE IF NOT EXISTS public.users (
                id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student' CHECK (role IN ('student', 'placement_team', 'admin')),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            -- Resumes table
            CREATE TABLE IF NOT EXISTS public.resumes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                file_name TEXT NOT NULL,
                file_url TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                raw_text TEXT,
                processed_text TEXT,
                extracted_skills JSONB DEFAULT '[]'::jsonb,
                extracted_experience JSONB DEFAULT '{}'::jsonb,
                uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            -- Job descriptions table
            CREATE TABLE IF NOT EXISTS public.job_descriptions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                uploader_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                company TEXT,
                file_name TEXT,
                file_url TEXT,
                raw_text TEXT NOT NULL,
                processed_text TEXT,
                job_title TEXT,
                must_have_skills JSONB DEFAULT '[]'::jsonb,
                good_to_have_skills JSONB DEFAULT '[]'::jsonb,
                qualifications JSONB DEFAULT '[]'::jsonb,
                experience_required TEXT,
                location TEXT,
                job_type TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            -- Evaluations table
            CREATE TABLE IF NOT EXISTS public.evaluations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                resume_id UUID NOT NULL REFERENCES public.resumes(id) ON DELETE CASCADE,
                jd_id UUID NOT NULL REFERENCES public.job_descriptions(id) ON DELETE CASCADE,
                relevance_score FLOAT NOT NULL CHECK (relevance_score >= 0 AND relevance_score <= 100),
                verdict TEXT NOT NULL CHECK (verdict IN ('High', 'Medium', 'Low')),
                matched_skills JSONB DEFAULT '[]'::jsonb,
                missing_skills JSONB DEFAULT '[]'::jsonb,
                skill_match_percentage FLOAT,
                experience_match BOOLEAN DEFAULT FALSE,
                experience_gap TEXT,
                relevant_projects JSONB DEFAULT '[]'::jsonb,
                missing_projects JSONB DEFAULT '[]'::jsonb,
                feedback TEXT,
                recommendations JSONB DEFAULT '[]'::jsonb,
                hard_match_score FLOAT,
                soft_match_score FLOAT,
                processing_time FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(resume_id, jd_id)
            );
            """,
            """
            -- Evaluation logs table
            CREATE TABLE IF NOT EXISTS public.evaluation_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                evaluation_id UUID REFERENCES public.evaluations(id) ON DELETE CASCADE,
                step_name TEXT NOT NULL,
                step_status TEXT NOT NULL CHECK (step_status IN ('success', 'error', 'warning')),
                step_data JSONB DEFAULT '{}'::jsonb,
                execution_time FLOAT,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            -- Create indexes for better performance
            CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON public.resumes(user_id);
            CREATE INDEX IF NOT EXISTS idx_job_descriptions_uploader_id ON public.job_descriptions(uploader_id);
            CREATE INDEX IF NOT EXISTS idx_evaluations_resume_id ON public.evaluations(resume_id);
            CREATE INDEX IF NOT EXISTS idx_evaluations_jd_id ON public.evaluations(jd_id);
            CREATE INDEX IF NOT EXISTS idx_evaluations_relevance_score ON public.evaluations(relevance_score);
            CREATE INDEX IF NOT EXISTS idx_evaluation_logs_evaluation_id ON public.evaluation_logs(evaluation_id);
            """,
            """
            -- Create updated_at trigger function
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            """,
            """
            -- Create triggers for updated_at
            DROP TRIGGER IF EXISTS update_users_updated_at ON public.users;
            CREATE TRIGGER update_users_updated_at 
                BEFORE UPDATE ON public.users 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                
            DROP TRIGGER IF EXISTS update_job_descriptions_updated_at ON public.job_descriptions;
            CREATE TRIGGER update_job_descriptions_updated_at 
                BEFORE UPDATE ON public.job_descriptions 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            """
        ]
        
        logger.info("Database tables SQL generated. Execute these in Supabase SQL editor:")
        for i, sql in enumerate(sql_statements, 1):
            logger.info(f"SQL Statement {i}:\n{sql}\n")
    
    async def create_storage_buckets(self):
        """Create storage buckets for file uploads."""
        try:
            # Create resume files bucket
            resume_bucket = self.client.storage.create_bucket("resume-files", {
                "public": False,
                "file_size_limit": 10485760,  # 10MB
                "allowed_mime_types": [
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/msword",
                    "text/plain"
                ]
            })
            
            if resume_bucket.error:
                logger.warning(f"Resume bucket creation warning: {resume_bucket.error}")
            else:
                logger.info("Resume files bucket created successfully")
            
            # Create job description files bucket
            jd_bucket = self.client.storage.create_bucket("job-description-files", {
                "public": False,
                "file_size_limit": 10485760,  # 10MB
                "allowed_mime_types": [
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/msword",
                    "text/plain"
                ]
            })
            
            if jd_bucket.error:
                logger.warning(f"JD bucket creation warning: {jd_bucket.error}")
            else:
                logger.info("Job description files bucket created successfully")
                
        except Exception as e:
            logger.error(f"Storage bucket creation failed: {e}")
    
    async def setup_row_level_security(self):
        """Setup row level security policies."""
        rls_policies = [
            """
            -- Enable RLS on all tables
            ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
            ALTER TABLE public.resumes ENABLE ROW LEVEL SECURITY;
            ALTER TABLE public.job_descriptions ENABLE ROW LEVEL SECURITY;
            ALTER TABLE public.evaluations ENABLE ROW LEVEL SECURITY;
            ALTER TABLE public.evaluation_logs ENABLE ROW LEVEL SECURITY;
            """,
            """
            -- Users can read their own profile
            CREATE POLICY "Users can read own profile" ON public.users
                FOR SELECT USING (auth.uid() = id);
                
            -- Users can update their own profile
            CREATE POLICY "Users can update own profile" ON public.users
                FOR UPDATE USING (auth.uid() = id);
            """,
            """
            -- Students can read/write their own resumes
            CREATE POLICY "Students can manage own resumes" ON public.resumes
                FOR ALL USING (auth.uid() = user_id);
                
            -- Placement team and admins can read all resumes
            CREATE POLICY "Placement team can read resumes" ON public.resumes
                FOR SELECT USING (
                    EXISTS (
                        SELECT 1 FROM public.users 
                        WHERE id = auth.uid() 
                        AND role IN ('placement_team', 'admin')
                    )
                );
            """,
            """
            -- Job descriptions policies
            CREATE POLICY "Users can read job descriptions" ON public.job_descriptions
                FOR SELECT USING (true);
                
            CREATE POLICY "Placement team can manage job descriptions" ON public.job_descriptions
                FOR ALL USING (
                    EXISTS (
                        SELECT 1 FROM public.users 
                        WHERE id = auth.uid() 
                        AND role IN ('placement_team', 'admin')
                    )
                );
            """,
            """
            -- Evaluation policies
            CREATE POLICY "Users can read relevant evaluations" ON public.evaluations
                FOR SELECT USING (
                    EXISTS (
                        SELECT 1 FROM public.resumes 
                        WHERE id = resume_id AND user_id = auth.uid()
                    ) OR
                    EXISTS (
                        SELECT 1 FROM public.job_descriptions 
                        WHERE id = jd_id AND uploader_id = auth.uid()
                    ) OR
                    EXISTS (
                        SELECT 1 FROM public.users 
                        WHERE id = auth.uid() AND role = 'admin'
                    )
                );
                
            CREATE POLICY "Service role can manage evaluations" ON public.evaluations
                FOR ALL USING (true);
            """
        ]
        
        logger.info("Row Level Security policies SQL generated. Execute these in Supabase SQL editor:")
        for i, policy in enumerate(rls_policies, 1):
            logger.info(f"RLS Policy {i}:\n{policy}\n")
    
    async def create_initial_data(self):
        """Create initial system data."""
        try:
            # Create admin user if not exists
            admin_data = {
                "email": "admin@resumechecker.com",
                "name": "System Administrator",
                "role": "admin",
                "is_active": True
            }
            
            # Note: In production, this should be done manually or via proper user creation
            logger.info("Create admin user manually in Supabase Auth, then add to users table:")
            logger.info(f"Admin data: {admin_data}")
            
        except Exception as e:
            logger.error(f"Initial data creation failed: {e}")


async def main():
    """Main function to run database initialization."""
    db_manager = DatabaseManager()
    await db_manager.initialize_database()


def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    return DatabaseManager()


if __name__ == "__main__":
    asyncio.run(main())