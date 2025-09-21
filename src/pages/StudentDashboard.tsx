import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Upload, FileText, Calendar, TrendingUp, Award, AlertCircle, Plus, Brain, LogOut, User, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import AISuggestionsComponent from "@/components/AISuggestions";
import { AISuggestionsResponse, apiClient } from "@/lib/api";

const StudentDashboard = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedResume, setSelectedResume] = useState<any>(null);
  const [aiSuggestions, setAiSuggestions] = useState<AISuggestionsResponse | null>(null);
  const [resumeText, setResumeText] = useState<string>("");
  const [targetRole, setTargetRole] = useState<string>("");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [authError, setAuthError] = useState<string>("");
  const [userResumes, setUserResumes] = useState<any[]>([]);
  const [isLoadingResumes, setIsLoadingResumes] = useState(false);
  const [uploadError, setUploadError] = useState<string>("");
  const [uploadSuccess, setUploadSuccess] = useState<string>("");
  
  // Job Description state
  const [userJobDescriptions, setUserJobDescriptions] = useState<any[]>([]);
  const [isLoadingJDs, setIsLoadingJDs] = useState(false);
  const [selectedJD, setSelectedJD] = useState<any>(null);
  const [jdUploadError, setJdUploadError] = useState<string>("");
  const [jdUploadSuccess, setJdUploadSuccess] = useState<string>("");
  const [jobTitle, setJobTitle] = useState<string>("");

 
  useEffect(() => {
    checkAuthStatus();
  }, []);


  useEffect(() => {
    if (isAuthenticated) {
      loadUserResumes();
      loadUserJobDescriptions();
    }
  }, [isAuthenticated]);

  const checkAuthStatus = async () => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      apiClient.setToken(token);
      try {
        const response = await apiClient.getCurrentUser();
        if (response.status === 'success') {
          setIsAuthenticated(true);
          setCurrentUser(response.data);
          setAuthError("");
        } else {
          // Token is invalid, clear it
          apiClient.clearToken();
          setIsAuthenticated(false);
          setAuthError("Please login again");
        }
      } catch (error) {
        apiClient.clearToken();
        setIsAuthenticated(false);
        setAuthError("Authentication check failed");
      }
    }
  };

  const loadUserResumes = async () => {
    setIsLoadingResumes(true);
    try {
      const response = await apiClient.getResumes();
      console.log('Resumes API response:', response); // Debug log
      if (response.status === 'success') {
        const resumesData = Array.isArray(response.data) ? response.data : [];
        console.log('Setting resumes data:', resumesData); // Debug log
        setUserResumes(resumesData);
      } else {
        console.error('Failed to load resumes:', response.message);
        setUserResumes([]); // Ensure it's always an array
        setAuthError(response.message || 'Failed to load resumes');
      }
    } catch (error) {
      console.error('Error loading resumes:', error);
      setUserResumes([]); // Ensure it's always an array
      setAuthError('Error loading resumes');
    } finally {
      setIsLoadingResumes(false);
    }
  };

  const loadUserJobDescriptions = async () => {
    setIsLoadingJDs(true);
    try {
      const response = await apiClient.getJobDescriptions();
      console.log('Job descriptions API response:', response); // Debug log
      if (response.status === 'success') {
        const jdsData = Array.isArray(response.data) ? response.data : [];
        console.log('Setting job descriptions data:', jdsData); // Debug log
        setUserJobDescriptions(jdsData);
      } else {
        console.error('Failed to load job descriptions:', response.message);
        setUserJobDescriptions([]);
        setAuthError(response.message || 'Failed to load job descriptions');
      }
    } catch (error) {
      console.error('Error loading job descriptions:', error);
      setUserJobDescriptions([]);
      setAuthError('Error loading job descriptions');
    } finally {
      setIsLoadingJDs(false);
    }
  };

  const handleLogout = () => {
    apiClient.clearToken();
    setIsAuthenticated(false);
    setCurrentUser(null);
    setAuthError("");
    setUserResumes([]);
    setUserJobDescriptions([]);
  };

  const handleFileUpload = async (file: File) => {
    if (!isAuthenticated) {
      setUploadError("Please login first to upload files");
      return;
    }

    setUploadError("");
    setUploadSuccess("");
    
    try {
      const response = await apiClient.uploadResume(file);
      if (response.status === 'success') {
        setUploadSuccess(`Successfully uploaded ${file.name}`);
        await loadUserResumes(); 
      } else {
        setUploadError(response.message || 'Upload failed');
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Upload error');
    }
  };

  const handleJDUpload = async (file: File, title: string) => {
    if (!isAuthenticated) {
      setJdUploadError("Please login first to upload job descriptions");
      return;
    }

    if (!title.trim()) {
      setJdUploadError("Please enter a job title");
      return;
    }

    setJdUploadError("");
    setJdUploadSuccess("");
    
    try {
      const response = await apiClient.uploadJobDescription(file, title);
      if (response.status === 'success') {
        setJdUploadSuccess(`Successfully uploaded job description: ${title}`);
        await loadUserJobDescriptions();
        setJobTitle(""); // Clear the input
        // Clear upload messages after 3 seconds
        setTimeout(() => {
          setJdUploadSuccess("");
        }, 3000);
      } else {
        setJdUploadError(response.message || 'Job description upload failed');
      }
    } catch (error) {
      setJdUploadError(error instanceof Error ? error.message : 'Job description upload error');
    }
  };

  // Sample resume text for demo purposes
  const sampleResumeText = `John Doe
Software Engineer

EXPERIENCE
Software Developer at Tech Corp (2022-2024)
- Developed web applications using React and Node.js
- Collaborated with cross-functional teams to deliver high-quality software
- Implemented responsive design principles for mobile compatibility

EDUCATION
Bachelor of Science in Computer Science
University of Technology (2018-2022)

SKILLS
- Programming: JavaScript, Python, Java
- Web Technologies: HTML, CSS, React, Node.js
- Databases: MySQL, MongoDB
- Tools: Git, VS Code, Postman

PROJECTS
E-commerce Website
- Built a full-stack e-commerce platform using MERN stack
- Integrated payment gateway and user authentication
- Deployed on AWS with CI/CD pipeline`;

  const handleAISuggestionsGenerated = (suggestions: AISuggestionsResponse) => {
    setAiSuggestions(suggestions);
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case "High":
        return "bg-success text-success-foreground";
      case "Medium":
        return "bg-warning text-warning-foreground";
      case "Low":
        return "bg-destructive text-destructive-foreground";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen pt-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <div className="flex justify-between items-start mb-4">
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-2">
                Student Dashboard
              </h1>
              <p className="text-muted-foreground text-lg">
                Upload and analyze your resumes to improve your job prospects
              </p>
            </div>
            
            {/* Authentication Status */}
            <div className="flex items-center space-x-4">
              {isAuthenticated ? (
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-2 bg-green-50 dark:bg-green-950/20 px-3 py-2 rounded-lg">
                    <User className="w-4 h-4 text-green-600" />
                    <span className="text-sm text-green-700 dark:text-green-300">
                      {currentUser?.name || 'Authenticated'}
                    </span>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleLogout}
                    className="text-red-600 border-red-200 hover:bg-red-50"
                  >
                    <LogOut className="w-4 h-4 mr-1" />
                    Logout
                  </Button>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <Alert className="py-2 px-3">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-sm">
                      Please login to access all features
                    </AlertDescription>
                  </Alert>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => window.location.href = '/login'}
                  >
                    Login
                  </Button>
                </div>
              )}
            </div>
          </div>
          
          {/* Auth Error */}
          {authError && (
            <Alert className="mb-4 border-red-200 bg-red-50">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{authError}</AlertDescription>
            </Alert>
          )}
          
          {/* Debug panel removed for production */}
        </motion.div>

        {/* Main Content Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <Tabs defaultValue="overview" className="space-y-6">
                        <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:grid-cols-3">
              <TabsTrigger value="overview" className="flex items-center space-x-2">
                <FileText className="w-4 h-4" />
                <span className="hidden sm:inline">Overview</span>
              </TabsTrigger>
              <TabsTrigger value="job-descriptions" className="flex items-center space-x-2">
                <Award className="w-4 h-4" />
                <span className="hidden sm:inline">Job Descriptions</span>
              </TabsTrigger>
              <TabsTrigger value="ai-analysis" className="flex items-center space-x-2">
                <Brain className="w-4 h-4" />
                <span className="hidden sm:inline">AI Analysis</span>
              </TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
            {/* Upload Section */}
            {/* Upload Messages */}
            {uploadError && (
              <Alert className="mb-4 border-red-200 bg-red-50">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription className="text-red-800">{uploadError}</AlertDescription>
              </Alert>
            )}
            
            {uploadSuccess && (
              <Alert className="mb-4 border-green-200 bg-green-50">
                <CheckCircle className="h-4 w-4" />
                <AlertDescription className="text-green-800">{uploadSuccess}</AlertDescription>
              </Alert>
            )}
            
            <div
                className={`card-premium border-2 border-dashed transition-all duration-300 ${
                  dragActive 
                    ? "border-primary bg-primary/5 scale-105" 
                    : "border-muted-foreground/30 hover:border-primary/50"
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Upload Your Resume</h3>
                  <p className="text-muted-foreground mb-6">
                    Drag and drop your resume here, or click to browse
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                    <input
                      type="file"
                      id="file-upload"
                      accept=".pdf,.docx,.doc"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                    <Button 
                      className="btn-hero"
                      onClick={() => document.getElementById('file-upload')?.click()}
                      disabled={!isAuthenticated}
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      {isAuthenticated ? 'Choose File' : 'Login to Upload'}
                    </Button>
                    <Button 
                      variant="outline"
                      onClick={() => {
                        setResumeText(sampleResumeText);
                        setTargetRole("Software Engineer");
                      }}
                    >
                      Use Sample Resume
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground mt-4">
                    Supports PDF, DOCX files up to 10MB
                  </p>
                </div>
              </div>              {/* Stats Cards */}
              <div className="grid md:grid-cols-3 gap-6">
                <Card className="card-premium">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Total Resumes</p>
                        <p className="text-2xl font-bold text-foreground">
                          {isLoadingResumes ? '...' : (userResumes || []).length}
                        </p>
                      </div>
                      <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
                        <FileText className="w-6 h-6 text-primary" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="card-premium">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Uploaded Today</p>
                        <p className="text-2xl font-bold text-foreground">
                          {(userResumes || []).filter(r => 
                            new Date(r.uploaded_at).toDateString() === new Date().toDateString()
                          ).length}
                        </p>
                      </div>
                      <div className="w-12 h-12 bg-success/10 rounded-xl flex items-center justify-center">
                        <TrendingUp className="w-6 h-6 text-success" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="card-premium">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Status</p>
                        <p className="text-lg font-bold text-foreground">
                          {isAuthenticated ? 'Ready' : 'Login Required'}
                        </p>
                      </div>
                      <div className="w-12 h-12 bg-warning/10 rounded-xl flex items-center justify-center">
                        <Award className="w-6 h-6 text-warning" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Resumes Table */}
              <Card className="card-premium">
                <CardHeader>
                  <CardTitle>Your Resumes</CardTitle>
                  <CardDescription>
                    {isLoadingResumes ? 'Loading your resumes...' : `You have ${(userResumes || []).length} resume(s) uploaded`}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoadingResumes ? (
                    <div className="flex items-center justify-center py-8">
                      <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                      <span className="ml-2 text-muted-foreground">Loading resumes...</span>
                    </div>
                  ) : (userResumes || []).length === 0 ? (
                    <div className="text-center py-8">
                      <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-2" />
                      <p className="text-muted-foreground">No resumes uploaded yet</p>
                      <p className="text-sm text-muted-foreground">Upload your first resume to get started!</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {(userResumes || []).map((resume, index) => (
                        <motion.div
                          key={resume.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.4, delay: index * 0.1 }}
                          className="glass-subtle rounded-xl p-6 hover:bg-muted/30 transition-all duration-300"
                        >
                          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                            <div className="flex items-center space-x-4">
                              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
                                <FileText className="w-6 h-6 text-primary" />
                              </div>
                              <div>
                                <h3 className="font-semibold text-foreground">{resume.filename}</h3>
                                <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-1">
                                  <div className="flex items-center space-x-1">
                                    <Calendar className="w-4 h-4" />
                                    <span>{new Date(resume.uploaded_at).toLocaleDateString()}</span>
                                  </div>
                                  <Badge variant="outline" className="text-xs">
                                    {(resume.file_size / 1024).toFixed(1)} KB
                                  </Badge>
                                </div>
                              </div>
                            </div>

                            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => {
                                  setResumeText(resume.extracted_text);
                                  setTargetRole("Software Engineer");
                                }}
                              >
                                <Brain className="w-4 h-4 mr-1" />
                                Analyze with AI
                              </Button>
                              <Button size="sm" className="btn-glass">
                                View Details
                              </Button>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Job Descriptions Tab */}
            <TabsContent value="job-descriptions" className="space-y-6">
              {/* JD Upload Messages */}
              {jdUploadError && (
                <Alert className="border-red-200 bg-red-50">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-red-800">{jdUploadError}</AlertDescription>
                </Alert>
              )}
              
              {jdUploadSuccess && (
                <Alert className="border-green-200 bg-green-50">
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription className="text-green-800">{jdUploadSuccess}</AlertDescription>
                </Alert>
              )}

              {/* JD Upload Section */}
              <Card className="card-premium">
                <CardHeader>
                  <CardTitle>Upload Job Description</CardTitle>
                  <CardDescription>
                    Upload job descriptions to compare against your resumes
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="job-title" className="text-sm font-medium">
                      Job Title
                    </Label>
                    <Input
                      id="job-title"
                      placeholder="e.g., Software Engineer, Data Scientist"
                      value={jobTitle}
                      onChange={(e) => setJobTitle(e.target.value)}
                      className="input-premium"
                    />
                  </div>
                  
                  <div className="border-2 border-dashed border-muted-foreground/30 rounded-xl p-8 text-center">
                    <Award className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Upload Job Description</h3>
                    <p className="text-muted-foreground mb-4">
                      Drag and drop a JD file or click to browse
                    </p>
                    <input
                      type="file"
                      id="jd-upload"
                      accept=".pdf,.docx,.doc,.txt"
                      onChange={(e) => {
                        if (e.target.files && e.target.files[0] && jobTitle.trim()) {
                          handleJDUpload(e.target.files[0], jobTitle);
                        } else if (!jobTitle.trim()) {
                          setJdUploadError("Please enter a job title first");
                        }
                      }}
                      className="hidden"
                    />
                    <Button 
                      onClick={() => {
                        if (!jobTitle.trim()) {
                          setJdUploadError("Please enter a job title first");
                          return;
                        }
                        document.getElementById('jd-upload')?.click();
                      }}
                      disabled={!isAuthenticated}
                      className="btn-hero"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      {isAuthenticated ? 'Choose JD File' : 'Login to Upload'}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Job Descriptions List */}
              <Card className="card-premium">
                <CardHeader>
                  <CardTitle>Your Job Descriptions</CardTitle>
                  <CardDescription>
                    {isLoadingJDs ? 'Loading job descriptions...' : `You have ${(userJobDescriptions || []).length} job description(s) uploaded`}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoadingJDs ? (
                    <div className="flex items-center justify-center py-8">
                      <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                      <span className="ml-2 text-muted-foreground">Loading job descriptions...</span>
                    </div>
                  ) : (userJobDescriptions || []).length === 0 ? (
                    <div className="text-center py-8">
                      <Award className="w-12 h-12 text-muted-foreground mx-auto mb-2" />
                      <p className="text-muted-foreground">No job descriptions uploaded yet</p>
                      <p className="text-sm text-muted-foreground">Upload job descriptions to compare with your resumes!</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {(userJobDescriptions || []).map((jd, index) => (
                        <motion.div
                          key={jd.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.4, delay: index * 0.1 }}
                          className="glass-subtle rounded-xl p-6 hover:bg-muted/30 transition-all duration-300"
                        >
                          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                            <div className="flex items-center space-x-4">
                              <div className="w-12 h-12 bg-warning/10 rounded-xl flex items-center justify-center">
                                <Award className="w-6 h-6 text-warning" />
                              </div>
                              <div>
                                <h3 className="font-semibold text-foreground">{jd.title}</h3>
                                <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-1">
                                  <div className="flex items-center space-x-1">
                                    <Calendar className="w-4 h-4" />
                                    <span>{new Date(jd.created_at).toLocaleDateString()}</span>
                                  </div>
                                  {jd.company && (
                                    <Badge variant="outline" className="text-xs">
                                      {jd.company}
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </div>

                            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => {
                                  setSelectedJD(jd);
                                  setTargetRole(jd.title);
                                }}
                              >
                                <Brain className="w-4 h-4 mr-1" />
                                Select for Analysis
                              </Button>
                              <Button size="sm" className="btn-glass">
                                View Details
                              </Button>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* AI Analysis Tab */}
            <TabsContent value="ai-analysis" className="space-y-6">
              {/* Resume & JD Selection */}
              <Card className="card-premium">
                <CardHeader>
                  <CardTitle>Resume vs Job Description Analysis</CardTitle>
                  <CardDescription>
                    Select a resume and job description to get detailed compatibility analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Resume Selection */}
                    <div className="space-y-4">
                      <Label className="text-sm font-medium">Select Resume</Label>
                      {(userResumes || []).length === 0 ? (
                        <div className="text-center py-4 text-muted-foreground">
                          <FileText className="w-8 h-8 mx-auto mb-2" />
                          <p>No resumes uploaded yet</p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {(userResumes || []).map((resume) => (
                            <div
                              key={resume.id}
                              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                                selectedResume?.id === resume.id
                                  ? 'border-primary bg-primary/5'
                                  : 'border-muted hover:border-primary/50'
                              }`}
                              onClick={() => {
                                setSelectedResume(resume);
                                setResumeText(resume.extracted_text);
                              }}
                            >
                              <p className="font-medium">{resume.filename}</p>
                              <p className="text-sm text-muted-foreground">
                                {new Date(resume.uploaded_at).toLocaleDateString()}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Job Description Selection */}
                    <div className="space-y-4">
                      <Label className="text-sm font-medium">Select Job Description</Label>
                      {(userJobDescriptions || []).length === 0 ? (
                        <div className="text-center py-4 text-muted-foreground">
                          <Award className="w-8 h-8 mx-auto mb-2" />
                          <p>No job descriptions uploaded yet</p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {(userJobDescriptions || []).map((jd) => (
                            <div
                              key={jd.id}
                              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                                selectedJD?.id === jd.id
                                  ? 'border-primary bg-primary/5'
                                  : 'border-muted hover:border-primary/50'
                              }`}
                              onClick={() => {
                                setSelectedJD(jd);
                                setTargetRole(jd.title);
                              }}
                            >
                              <p className="font-medium">{jd.title}</p>
                              <p className="text-sm text-muted-foreground">
                                {new Date(jd.created_at).toLocaleDateString()}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Analysis Button */}
                  <div className="flex justify-center">
                    <Button
                      className="btn-hero"
                      disabled={!selectedResume || (!selectedJD && !targetRole)}
                      onClick={() => {
                        if (selectedResume && (selectedJD || targetRole)) {
                          // Trigger AI analysis
                          setAiSuggestions(null); // Reset previous suggestions
                        }
                      }}
                    >
                      <Brain className="w-4 h-4 mr-2" />
                      Analyze Resume Relevance
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* AI Analysis Results */}
              <AISuggestionsComponent
                resumeText={resumeText}
                targetRole={targetRole}
                onSuggestionsGenerated={handleAISuggestionsGenerated}
              />
            </TabsContent>

            {/* Suggestions tab removed to avoid duplication with AI Analysis */}
          </Tabs>
        </motion.div>
      </div>
    </div>
  );
};

export default StudentDashboard;