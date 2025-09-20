# Resume Relevance Check System - Backend

A production-ready FastAPI backend for an AI-powered resume relevance checking system with Supabase integration.

## Features

- **Authentication & Authorization**: JWT-based auth with role-based access control (Student/Placement Team/Admin)
- **File Processing**: Support for PDF, DOCX, and TXT resume/job description uploads
- **AI Evaluation Pipeline**: Multi-stage evaluation using keyword matching, semantic similarity, and LLM analysis
- **Comprehensive APIs**: REST APIs for all operations with proper error handling
- **Observability**: LangSmith integration for LLM monitoring and application metrics
- **Scalable Architecture**: Designed for serverless deployment on Vercel/Railway

## Tech Stack

- **Framework**: FastAPI
- **Database**: Supabase (PostgreSQL)
- **Storage**: Supabase Storage
- **AI/ML**: 
  - LangChain & LangGraph for LLM workflows
  - OpenAI GPT/Anthropic Claude for advanced analysis
  - scikit-learn for TF-IDF similarity
  - spaCy for NLP processing
- **File Processing**: PyMuPDF, pdfplumber, python-docx
- **Monitoring**: LangSmith for LLM observability

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Configuration and settings
│   ├── core/                  # Core utilities
│   │   ├── auth.py            # Authentication & JWT handling
│   │   ├── supabase.py        # Supabase client & storage
│   │   ├── file_processing.py # File parsing & text extraction
│   │   ├── evaluation.py      # AI evaluation pipeline
│   │   ├── observability.py   # LangSmith & metrics
│   │   ├── logging.py         # Logging configuration
│   │   └── database_init.py   # Database setup utilities
│   ├── api/v1/                # API endpoints
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── resume.py          # Resume management
│   │   ├── job_description.py # Job description management
│   │   ├── evaluation.py      # Evaluation endpoints
│   │   ├── dashboard.py       # Dashboard APIs
│   │   └── router.py          # Main API router
│   ├── models/                # Database models
│   │   └── database.py        # SQLAlchemy models
│   └── schemas/               # Pydantic schemas
│       └── __init__.py        # Request/response models
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY`: Supabase service role key
- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `OPENAI_API_KEY`: OpenAI API key (optional, for advanced AI)
- `LANGCHAIN_API_KEY`: LangSmith API key (optional, for observability)

### 3. Database Setup

1. Create a new Supabase project
2. Run the SQL statements from `app/core/database_init.py` in Supabase SQL editor
3. Create storage buckets for file uploads

### 4. Run the Application

```bash
# Development
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

### Authentication
- `POST /api/v1/auth/signup` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout
- `GET /api/v1/auth/me` - Get current user info

### Resume Management
- `POST /api/v1/resume/upload` - Upload resume file
- `GET /api/v1/resume/list` - List user's resumes
- `GET /api/v1/resume/{resume_id}` - Get resume details
- `DELETE /api/v1/resume/{resume_id}` - Delete resume

### Job Description Management
- `POST /api/v1/jd/upload` - Upload job description
- `GET /api/v1/jd/list` - List job descriptions
- `GET /api/v1/jd/{jd_id}` - Get job description details
- `PUT /api/v1/jd/{jd_id}` - Update job description
- `DELETE /api/v1/jd/{jd_id}` - Delete job description

### Evaluation
- `POST /api/v1/evaluate/` - Evaluate resume against job description
- `GET /api/v1/evaluate/{evaluation_id}` - Get evaluation results
- `GET /api/v1/evaluate/resume/{resume_id}` - Get all evaluations for resume
- `GET /api/v1/evaluate/jd/{jd_id}` - Get all evaluations for job description

### Dashboard
- `GET /api/v1/dashboard/student` - Student dashboard
- `GET /api/v1/dashboard/placement/{jd_id}` - Placement team dashboard
- `GET /api/v1/dashboard/analytics` - Analytics data

## Evaluation Pipeline

The system uses a multi-stage evaluation approach:

### 1. Hard Match (40% weight)
- Keyword-based skill matching
- Fuzzy string matching for skill variations
- Synonym detection for technology terms

### 2. Soft Match (30% weight)
- TF-IDF vectorization and cosine similarity
- Optional: OpenAI embeddings for semantic similarity

### 3. LLM Analysis (30% weight)
- Advanced analysis using GPT/Claude
- Experience gap identification
- Project relevance assessment
- Personalized feedback generation

### Final Output
- Relevance Score (0-100)
- Verdict (High/Medium/Low)
- Matched/Missing skills
- Experience analysis
- Improvement recommendations

## Deployment

### Vercel Deployment

1. Install Vercel CLI: `npm install -g vercel`
2. Create `vercel.json`:

```json
{
  "builds": [
    {
      "src": "app/main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app/main.py"
    }
  ]
}
```

3. Deploy: `vercel --prod`

### Railway Deployment

1. Connect GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on git push

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring & Observability

### LangSmith Integration
- Automatic tracing of LLM calls
- Evaluation performance metrics
- Error tracking and debugging

### Application Metrics
- Evaluation success/failure rates
- Processing time monitoring
- File upload statistics
- User activity tracking

### Health Checks
- `GET /health` - Basic health check
- Database connectivity monitoring
- External API availability checks

## Security Features

- JWT-based authentication
- Role-based access control
- Row-level security in database
- File upload validation
- Rate limiting (configurable)
- CORS protection
- Input sanitization

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
# Formatting
black .

# Linting
flake8 .

# Type checking
mypy .
```

### Adding New Features
1. Create feature branch
2. Add API endpoints in `app/api/v1/`
3. Add business logic in `app/core/`
4. Update schemas in `app/schemas/`
5. Add tests
6. Update documentation

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check Supabase URL and keys
   - Verify database tables are created

2. **File Upload Fails**
   - Check storage bucket permissions
   - Verify file size limits

3. **AI Evaluation Errors**
   - Verify API keys for OpenAI/Anthropic
   - Check rate limits

4. **Authentication Issues**
   - Verify JWT secret key
   - Check token expiration settings

### Debugging

Enable debug logging:
```python
LOG_LEVEL=DEBUG
```

Check LangSmith dashboard for LLM call traces and errors.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create GitHub issue
- Check documentation
- Review API docs at `/docs` endpoint