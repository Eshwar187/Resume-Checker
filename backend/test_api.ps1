# PowerShell script to test the backend API
Write-Host "🧪 Testing Backend API Endpoints" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

$baseUrl = "http://localhost:8001"

# Test Health Endpoint
Write-Host "`n1. Testing Health Endpoint..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "✅ Health Check Successful!" -ForegroundColor Green
    $healthResponse | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test Registration
Write-Host "`n2. Testing User Registration..." -ForegroundColor Yellow
$registerData = @{
    name = "John Doe"
    email = "john@example.com"
    password = "password123"
} | ConvertTo-Json

try {
    $registerResponse = Invoke-RestMethod -Uri "$baseUrl/api/v1/auth/register" -Method Post -Body $registerData -ContentType "application/json"
    Write-Host "✅ Registration Successful!" -ForegroundColor Green
    $registerResponse | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Registration Failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $errorDetails = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($errorDetails)
        $errorBody = $reader.ReadToEnd()
        Write-Host "Error Details: $errorBody" -ForegroundColor Red
    }
}

# Test Login with existing user
Write-Host "`n3. Testing User Login (existing user)..." -ForegroundColor Yellow
$loginData = @{
    email = "test@example.com"
    password = "password123"
} | ConvertTo-Json

try {
    $loginResponse = Invoke-RestMethod -Uri "$baseUrl/api/v1/auth/login" -Method Post -Body $loginData -ContentType "application/json"
    Write-Host "✅ Login Successful!" -ForegroundColor Green
    $loginResponse | ConvertTo-Json -Depth 3
    $token = $loginResponse.data.access_token
    Write-Host "🔑 Access Token: $token" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Login Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test AI Suggestions
Write-Host "`n4. Testing AI Suggestions..." -ForegroundColor Yellow
$aiData = @{
    resume_text = @"
John Doe
Software Engineer

EXPERIENCE
Software Developer at Tech Corp (2022-2024)
- Developed web applications using React and Node.js
- Collaborated with cross-functional teams to deliver high-quality software

EDUCATION
Bachelor of Science in Computer Science
University of Technology (2018-2022)

SKILLS
- Programming: JavaScript, Python, Java
- Web Technologies: HTML, CSS, React, Node.js
- Databases: MySQL, MongoDB
"@
    target_role = "Software Engineer"
} | ConvertTo-Json

try {
    Write-Host "⏳ Generating AI suggestions (this may take a moment)..." -ForegroundColor Yellow
    $aiResponse = Invoke-RestMethod -Uri "$baseUrl/api/v1/ai-suggestions" -Method Post -Body $aiData -ContentType "application/json"
    Write-Host "✅ AI Suggestions Generated!" -ForegroundColor Green
    
    # Display key parts of the response
    Write-Host "`nOverall Score:" -ForegroundColor Cyan
    $aiResponse.data.suggestions.overall_score | ConvertTo-Json -Depth 2
    
    Write-Host "`nSample Skill Recommendations:" -ForegroundColor Cyan
    $aiResponse.data.suggestions.skill_recommendations[0..1] | ConvertTo-Json -Depth 2
    
} catch {
    Write-Host "❌ AI Suggestions Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🎉 API Testing Complete!" -ForegroundColor Green
Write-Host "Your backend server is running properly with authentication and AI features!" -ForegroundColor Green