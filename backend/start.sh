#!/bin/bash
# Render.com start script for Resume Checker Backend

echo "🚀 Starting Resume Checker Backend server..."

# Set default port if not provided
export PORT=${PORT:-8000}

# Start the FastAPI application with Uvicorn
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --log-level info