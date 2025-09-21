import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, FileText, Calendar, TrendingUp, Award, AlertCircle, Plus, Brain, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import AISuggestionsComponent from "@/components/AISuggestions";
import { AISuggestionsResponse } from "@/lib/api";

const StudentDashboard = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedResume, setSelectedResume] = useState<any>(null);
  const [aiSuggestions, setAiSuggestions] = useState<AISuggestionsResponse | null>(null);
  const [resumeText, setResumeText] = useState<string>("");
  const [targetRole, setTargetRole] = useState<string>("");

  // Sample resume text for demo purposes - in production this would be extracted from uploaded files
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

  // Mock data - will be replaced with Supabase data
  const resumes = [
    {
      id: 1,
      name: "Software Engineer Resume.pdf",
      uploadDate: "2024-01-15",
      status: "analyzed",
      relevanceScore: 87,
      verdict: "High",
      missingSkills: ["Docker", "Kubernetes"],
      matchedSkills: ["React", "Node.js", "Python", "SQL"],
    },
    {
      id: 2,
      name: "Data Analyst Resume.pdf",
      uploadDate: "2024-01-10",
      status: "analyzed",
      relevanceScore: 72,
      verdict: "Medium",
      missingSkills: ["Tableau", "R Programming"],
      matchedSkills: ["Python", "SQL", "Excel"],
    },
    {
      id: 3,
      name: "Marketing Resume.pdf",
      uploadDate: "2024-01-08",
      status: "processing",
      relevanceScore: null,
      verdict: null,
      missingSkills: [],
      matchedSkills: [],
    },
  ];

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
      // Handle file upload - will be connected to Supabase
      console.log("File dropped:", e.dataTransfer.files[0]);
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
          <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-2">
            Student Dashboard
          </h1>
          <p className="text-muted-foreground text-lg">
            Upload and analyze your resumes to improve your job prospects
          </p>
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
                <span>Overview</span>
              </TabsTrigger>
              <TabsTrigger value="ai-analysis" className="flex items-center space-x-2">
                <Brain className="w-4 h-4" />
                <span>AI Analysis</span>
              </TabsTrigger>
              <TabsTrigger value="suggestions" className="flex items-center space-x-2">
                <Sparkles className="w-4 h-4" />
                <span>Suggestions</span>
              </TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
              {/* Upload Section */}
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
                    <Button 
                      className="btn-hero"
                      onClick={() => {
                        setResumeText(sampleResumeText);
                        setTargetRole("Software Engineer");
                      }}
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Use Sample Resume
                    </Button>
                    <p className="text-sm text-muted-foreground">
                      Supports PDF, DOCX files up to 10MB
                    </p>
                  </div>
                </div>
              </div>

              {/* Stats Cards */}
              <div className="grid md:grid-cols-3 gap-6">
                <Card className="card-premium">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Total Resumes</p>
                        <p className="text-2xl font-bold text-foreground">{resumes.length}</p>
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
                        <p className="text-sm font-medium text-muted-foreground">Average Score</p>
                        <p className="text-2xl font-bold text-foreground">
                          {Math.round(resumes.filter(r => r.relevanceScore).reduce((acc, r) => acc + r.relevanceScore!, 0) / resumes.filter(r => r.relevanceScore).length)}%
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
                        <p className="text-sm font-medium text-muted-foreground">High Relevance</p>
                        <p className="text-2xl font-bold text-foreground">
                          {resumes.filter(r => r.verdict === "High").length}
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
                    Track and analyze all your uploaded resumes
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {resumes.map((resume, index) => (
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
                              <h3 className="font-semibold text-foreground">{resume.name}</h3>
                              <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-1">
                                <div className="flex items-center space-x-1">
                                  <Calendar className="w-4 h-4" />
                                  <span>{new Date(resume.uploadDate).toLocaleDateString()}</span>
                                </div>
                                {resume.status === "processing" && (
                                  <div className="flex items-center space-x-1">
                                    <div className="w-2 h-2 bg-warning rounded-full animate-pulse" />
                                    <span>Processing...</span>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>

                          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                            {resume.status === "analyzed" && (
                              <>
                                <div className="flex items-center space-x-4">
                                  <div>
                                    <p className="text-sm text-muted-foreground">Relevance Score</p>
                                    <div className="flex items-center space-x-2">
                                      <Progress value={resume.relevanceScore!} className="w-20" />
                                      <span className="text-sm font-medium">{resume.relevanceScore}%</span>
                                    </div>
                                  </div>
                                  <Badge className={getVerdictColor(resume.verdict!)}>
                                    {resume.verdict}
                                  </Badge>
                                </div>
                                <div className="flex space-x-2">
                                  <Button size="sm" className="btn-glass">
                                    View Details
                                  </Button>
                                  <Button 
                                    size="sm" 
                                    variant="outline"
                                    onClick={() => {
                                      setSelectedResume(resume);
                                      setResumeText(sampleResumeText);
                                      setTargetRole("Software Engineer");
                                    }}
                                  >
                                    <Brain className="w-4 h-4 mr-1" />
                                    AI Analysis
                                  </Button>
                                </div>
                              </>
                            )}
                            
                            {resume.status === "processing" && (
                              <div className="flex items-center space-x-2">
                                <div className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                                <span className="text-sm text-muted-foreground">Analyzing...</span>
                              </div>
                            )}
                          </div>
                        </div>

                        {resume.status === "analyzed" && (
                          <div className="mt-4 pt-4 border-t border-muted/20">
                            <div className="grid md:grid-cols-2 gap-4">
                              <div>
                                <p className="text-sm font-medium text-foreground mb-2">Matched Skills</p>
                                <div className="flex flex-wrap gap-2">
                                  {resume.matchedSkills.map((skill) => (
                                    <Badge key={skill} variant="secondary" className="bg-success/10 text-success">
                                      {skill}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                              <div>
                                <p className="text-sm font-medium text-foreground mb-2">Missing Skills</p>
                                <div className="flex flex-wrap gap-2">
                                  {resume.missingSkills.map((skill) => (
                                    <Badge key={skill} variant="secondary" className="bg-warning/10 text-warning">
                                      <AlertCircle className="w-3 h-3 mr-1" />
                                      {skill}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* AI Analysis Tab */}
            <TabsContent value="ai-analysis">
              <AISuggestionsComponent
                resumeText={resumeText}
                targetRole={targetRole}
                onSuggestionsGenerated={handleAISuggestionsGenerated}
              />
            </TabsContent>

            {/* Suggestions Tab */}
            <TabsContent value="suggestions">
              {aiSuggestions ? (
                <AISuggestionsComponent
                  resumeText={resumeText}
                  targetRole={targetRole}
                  onSuggestionsGenerated={handleAISuggestionsGenerated}
                />
              ) : (
                <Card className="card-premium">
                  <CardContent className="p-12 text-center">
                    <div className="w-16 h-16 bg-muted/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <Sparkles className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <h3 className="text-xl font-semibold mb-2">No Suggestions Yet</h3>
                    <p className="text-muted-foreground mb-6">
                      Upload a resume and run AI analysis to get personalized suggestions
                    </p>
                    <Button 
                      className="btn-hero"
                      onClick={() => {
                        setResumeText(sampleResumeText);
                        setTargetRole("Software Engineer");
                      }}
                    >
                      <Brain className="w-4 h-4 mr-2" />
                      Try Sample Analysis
                    </Button>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        </motion.div>
      </div>
    </div>
  );
};

export default StudentDashboard;