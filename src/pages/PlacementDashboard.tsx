import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Filter, Upload, FileText, Users, TrendingUp, Eye, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";

const PlacementDashboard = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterVerdict, setFilterVerdict] = useState("all");
  const [filterRole, setFilterRole] = useState("all");

  // Mock data - will be replaced with Supabase data
  const jobDescriptions = [
    {
      id: 1,
      title: "Senior Software Engineer",
      company: "TechCorp",
      uploadDate: "2024-01-15",
      requirements: ["React", "Node.js", "Python", "Docker", "Kubernetes"],
      candidatesCount: 15,
    },
    {
      id: 2,
      title: "Data Analyst",
      company: "DataInc",
      uploadDate: "2024-01-12",
      requirements: ["Python", "SQL", "Tableau", "R Programming"],
      candidatesCount: 8,
    },
  ];

  const candidates = [
    {
      id: 1,
      name: "John Smith",
      email: "john.smith@email.com",
      jobRole: "Senior Software Engineer",
      relevanceScore: 87,
      verdict: "High",
      resumeName: "John_Smith_Resume.pdf",
      matchedSkills: ["React", "Node.js", "Python"],
      missingSkills: ["Docker", "Kubernetes"],
      uploadDate: "2024-01-15",
    },
    {
      id: 2,
      name: "Sarah Johnson",
      email: "sarah.j@email.com",
      jobRole: "Senior Software Engineer",
      relevanceScore: 72,
      verdict: "Medium",
      resumeName: "Sarah_Johnson_Resume.pdf",
      matchedSkills: ["React", "Python"],
      missingSkills: ["Node.js", "Docker", "Kubernetes"],
      uploadDate: "2024-01-14",
    },
    {
      id: 3,
      name: "Mike Chen",
      email: "mike.chen@email.com",
      jobRole: "Data Analyst",
      relevanceScore: 94,
      verdict: "High",
      resumeName: "Mike_Chen_Resume.pdf",
      matchedSkills: ["Python", "SQL", "Tableau"],
      missingSkills: ["R Programming"],
      uploadDate: "2024-01-13",
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

  const filteredCandidates = candidates.filter((candidate) => {
    const matchesSearch = candidate.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         candidate.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesVerdict = filterVerdict === "all" || candidate.verdict === filterVerdict;
    const matchesRole = filterRole === "all" || candidate.jobRole === filterRole;
    
    return matchesSearch && matchesVerdict && matchesRole;
  });

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
            Placement Team Dashboard
          </h1>
          <p className="text-muted-foreground text-lg">
            Manage job descriptions and evaluate candidate resumes
          </p>
        </motion.div>

        {/* Upload JD Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-8"
        >
          <Card className="card-premium">
            <CardHeader>
              <CardTitle>Upload Job Description</CardTitle>
              <CardDescription>
                Upload job descriptions to start matching candidates
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row gap-4">
                <Button className="btn-hero">
                  <Upload className="w-4 h-4 mr-2" />
                  Upload JD (PDF/DOCX)
                </Button>
                <Button variant="outline" className="btn-glass">
                  <FileText className="w-4 h-4 mr-2" />
                  Paste Text JD
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="grid md:grid-cols-4 gap-6 mb-8"
        >
          <Card className="card-premium">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Active JDs</p>
                  <p className="text-2xl font-bold text-foreground">{jobDescriptions.length}</p>
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
                  <p className="text-sm font-medium text-muted-foreground">Total Candidates</p>
                  <p className="text-2xl font-bold text-foreground">{candidates.length}</p>
                </div>
                <div className="w-12 h-12 bg-success/10 rounded-xl flex items-center justify-center">
                  <Users className="w-6 h-6 text-success" />
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
                    {candidates.filter(c => c.verdict === "High").length}
                  </p>
                </div>
                <div className="w-12 h-12 bg-warning/10 rounded-xl flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-warning" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-premium">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Avg Score</p>
                  <p className="text-2xl font-bold text-foreground">
                    {Math.round(candidates.reduce((acc, c) => acc + c.relevanceScore, 0) / candidates.length)}%
                  </p>
                </div>
                <div className="w-12 h-12 bg-destructive/10 rounded-xl flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-destructive" />
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Filters and Search */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mb-8"
        >
          <Card className="card-premium">
            <CardContent className="p-6">
              <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      placeholder="Search candidates by name or email..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="input-premium pl-10"
                    />
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row gap-4">
                  <Select value={filterRole} onValueChange={setFilterRole}>
                    <SelectTrigger className="input-premium w-48">
                      <SelectValue placeholder="Filter by role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Roles</SelectItem>
                      <SelectItem value="Senior Software Engineer">Senior Software Engineer</SelectItem>
                      <SelectItem value="Data Analyst">Data Analyst</SelectItem>
                    </SelectContent>
                  </Select>

                  <Select value={filterVerdict} onValueChange={setFilterVerdict}>
                    <SelectTrigger className="input-premium w-48">
                      <SelectValue placeholder="Filter by verdict" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Verdicts</SelectItem>
                      <SelectItem value="High">High Relevance</SelectItem>
                      <SelectItem value="Medium">Medium Relevance</SelectItem>
                      <SelectItem value="Low">Low Relevance</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Candidates Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <Card className="card-premium">
            <CardHeader>
              <CardTitle>Candidate Evaluations</CardTitle>
              <CardDescription>
                Review and manage candidate resume evaluations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {filteredCandidates.map((candidate, index) => (
                  <motion.div
                    key={candidate.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: index * 0.1 }}
                    className="glass-subtle rounded-xl p-6 hover:bg-muted/30 transition-all duration-300"
                  >
                    <div className="flex flex-col lg:flex-row justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <h3 className="font-semibold text-foreground text-lg">{candidate.name}</h3>
                            <p className="text-muted-foreground">{candidate.email}</p>
                          </div>
                          <Badge className={getVerdictColor(candidate.verdict)}>
                            {candidate.verdict} Relevance
                          </Badge>
                        </div>

                        <div className="grid md:grid-cols-2 gap-4 mb-4">
                          <div>
                            <p className="text-sm font-medium text-muted-foreground mb-2">Applied For</p>
                            <p className="text-foreground">{candidate.jobRole}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-muted-foreground mb-2">Resume</p>
                            <p className="text-foreground text-sm">{candidate.resumeName}</p>
                          </div>
                        </div>

                        <div className="mb-4">
                          <div className="flex items-center justify-between mb-2">
                            <p className="text-sm font-medium text-muted-foreground">Relevance Score</p>
                            <span className="text-sm font-medium text-foreground">{candidate.relevanceScore}%</span>
                          </div>
                          <Progress value={candidate.relevanceScore} className="h-2" />
                        </div>

                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm font-medium text-foreground mb-2">Matched Skills</p>
                            <div className="flex flex-wrap gap-2">
                              {candidate.matchedSkills.map((skill) => (
                                <Badge key={skill} variant="secondary" className="bg-success/10 text-success">
                                  {skill}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-foreground mb-2">Missing Skills</p>
                            <div className="flex flex-wrap gap-2">
                              {candidate.missingSkills.map((skill) => (
                                <Badge key={skill} variant="secondary" className="bg-warning/10 text-warning">
                                  {skill}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col sm:flex-row lg:flex-col gap-2 lg:w-48">
                        <Button size="sm" className="btn-hero">
                          <Eye className="w-4 h-4 mr-2" />
                          View Resume
                        </Button>
                        <Button size="sm" variant="outline" className="btn-glass">
                          <Download className="w-4 h-4 mr-2" />
                          Download
                        </Button>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default PlacementDashboard;