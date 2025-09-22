# Resume Checker Backend - Render Deployment

This FastAPI backend is configured for deployment on Render.com.

## Deployment Instructions

### 1. Push to Git Repository
Make sure your code is pushed to GitHub, GitLab, or Bitbucket.

### 2. Create New Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Select the repository: `Resume-Checker`
5. Configure the service:

**Basic Settings:**
- Name: `resume-checker-backend`
- Region: `Oregon` (or your preferred region)
- Branch: `main`
- Root Directory: `backend`
- Runtime: `Python 3`
- Build Command: `./build.sh`
- Start Command: `./start.sh`

**Environment Variables:**
Set these in the Render dashboard under "Environment":

```
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
DATABASE_URL=your-database-url
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
LANGCHAIN_API_KEY=your-langchain-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env
```

### 3. Deploy

1. Click "Create Web Service"
2. Render will automatically build and deploy
3. Your backend will be available at: `https://your-service-name.onrender.com`

### 4. Update Frontend

Update your frontend's `VITE_API_URL` environment variable to point to your Render backend URL.

## Health Check

Once deployed, test your backend:
- Health: `https://your-service-name.onrender.com/health`
- Docs: `https://your-service-name.onrender.com/docs` (if DEBUG=true)

## Features Included

- ✅ FastAPI with Uvicorn
- ✅ Auto-scaling with Render
- ✅ Environment-based configuration
- ✅ CORS setup for frontend integration
- ✅ Supabase database integration
- ✅ File upload support
- ✅ AI/LLM integrations (OpenAI, Anthropic, Google)
- ✅ Vector database support (Pinecone, ChromaDB, FAISS)
- ✅ Authentication with JWT
- ✅ Streaming endpoints for real-time AI

## Troubleshooting

**Build Fails:**
- Check that all dependencies are in `requirements.txt`
- Verify Python version compatibility
- Check build logs in Render dashboard

**App Won't Start:**
- Verify `start.sh` has correct permissions
- Check that `app.main:app` path is correct
- Review startup logs

**CORS Issues:**
- Add your frontend domain to `allowed_origins` in `app/config.py`
- Redeploy after making changes

**Database Connection:**
- Verify DATABASE_URL format
- Check Supabase credentials
- Ensure database is accessible from Render IPs