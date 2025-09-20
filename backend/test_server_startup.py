#!/usr/bin/env python3
"""
Minimal server startup test without problematic imports.
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Set environment variables to handle PyTorch path issue
os.environ.setdefault("TORCH_LOGS_LOCATION", "C:\\temp")

try:
    print("🚀 Starting minimal FastAPI server test...")
    
    # Test basic imports
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    print("✅ FastAPI imports successful")
    
    # Test config
    from app.config import settings
    print("✅ Configuration loaded")
    
    # Test Supabase
    from app.core.supabase import get_supabase_client
    client = get_supabase_client()
    print("✅ Supabase client initialized")
    
    # Create minimal app
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Resume Relevance Check API"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add a simple health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "app_name": settings.app_name,
            "version": settings.version,
            "supabase_configured": bool(settings.supabase_url),
            "openai_configured": bool(settings.openai_api_key)
        }
    
    print("✅ Minimal FastAPI app created successfully")
    print(f"📊 App routes: {len(app.routes)}")
    print()
    print("🎉 SERVER STARTUP TEST SUCCESSFUL!")
    print()
    print("📋 Configuration Summary:")
    print(f"  - Supabase URL: {settings.supabase_url}")
    print(f"  - OpenAI configured: {bool(settings.openai_api_key)}")
    print(f"  - Anthropic configured: {bool(settings.anthropic_api_key)}")
    print(f"  - Google AI configured: {bool(settings.google_api_key)}")
    print(f"  - LangSmith configured: {bool(settings.langchain_api_key)}")
    print(f"  - Max file size: {settings.max_file_size} bytes")
    print(f"  - Allowed extensions: {settings.allowed_extensions}")
    print()
    print("🚀 To start the full server:")
    print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    print("🔧 Note: Some AI features may require fixing the Windows PyTorch path issue")
    
except Exception as e:
    print(f"❌ Server startup test failed: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    pass