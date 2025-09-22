#!/bin/bash
# Render.com build script for Resume Checker Backend

echo "🚀 Starting Resume Checker Backend build..."

# Update pip
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy English model
echo "🔤 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Verify installation
echo "✅ Verifying installation..."
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo "✅ Build completed successfully!"