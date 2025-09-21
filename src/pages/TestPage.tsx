import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  User, 
  LogIn, 
  LogOut, 
  Upload, 
  Brain, 
  CheckCircle, 
  AlertCircle,
  Loader
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/lib/api';

const TestPage = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  
  // Auth form data
  const [authForm, setAuthForm] = useState({
    name: 'Test User',
    email: 'test@example.com',
    password: 'password123'
  });

  // Resume analysis data
  const [resumeText, setResumeText] = useState(`John Doe
Software Engineer

EDUCATION
Bachelor of Science in Computer Science
University of Technology (2018-2022)

SKILLS
- Programming: JavaScript, Python, Java
- Web Technologies: HTML, CSS, React, Node.js
- Databases: MySQL, MongoDB

EXPERIENCE
Software Developer at Tech Corp (2022-2024)
- Developed web applications using React and Node.js
- Collaborated with cross-functional teams to deliver high-quality software
- Implemented responsive design principles for mobile compatibility

PROJECTS
E-commerce Website
- Built a full-stack e-commerce platform using MERN stack
- Integrated payment gateway and user authentication
- Deployed on AWS with CI/CD pipeline`);

  const [targetRole, setTargetRole] = useState('Software Engineer');
  const [aiResults, setAiResults] = useState<any>(null);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      apiClient.setToken(token);
      try {
        const response = await apiClient.getCurrentUser();
        if (response.status === 'success') {
          setIsAuthenticated(true);
          setCurrentUser(response.data);
        } else {
          apiClient.clearToken();
          setIsAuthenticated(false);
        }
      } catch (error) {
        apiClient.clearToken();
        setIsAuthenticated(false);
      }
    }
  };

  const handleRegister = async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    
    try {
      const response = await apiClient.register(authForm.email, authForm.password, authForm.name);
      
      if (response.status === 'success' && response.data) {
        apiClient.setToken(response.data.access_token);
        setSuccess('Registration successful!');
        await checkAuthStatus();
      } else {
        setError(response.message || 'Registration failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration error');
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    
    try {
      const response = await apiClient.login(authForm.email, authForm.password);
      
      if (response.status === 'success' && response.data) {
        apiClient.setToken(response.data.access_token);
        setSuccess('Login successful!');
        await checkAuthStatus();
      } else {
        setError(response.message || 'Login failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login error');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    apiClient.clearToken();
    setIsAuthenticated(false);
    setCurrentUser(null);
    setSuccess('Logged out successfully');
    setError('');
  };

  const handleAIAnalysis = async () => {
    setLoading(true);
    setError('');
    setAiResults(null);
    
    try {
      const response = await apiClient.getAISuggestions(resumeText, targetRole);
      
      if (response.status === 'success') {
        setAiResults(response.data);
        setSuccess('AI analysis completed!');
      } else {
        setError(response.message || 'AI analysis failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'AI analysis error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-2">
            🧪 API Test Page
          </h1>
          <p className="text-muted-foreground text-lg">
            Test authentication, file upload, and AI analysis features
          </p>
        </motion.div>

        {/* Messages */}
        {error && (
          <Alert className="mb-6 border-red-200 bg-red-50">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}
        
        {success && (
          <Alert className="mb-6 border-green-200 bg-green-50">
            <CheckCircle className="h-4 w-4" />
            <AlertDescription className="text-green-800">{success}</AlertDescription>
          </Alert>
        )}

        {/* Authentication Status */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <User className="w-5 h-5" />
              <span>Authentication Status</span>
              {isAuthenticated ? (
                <Badge className="bg-green-100 text-green-800">Authenticated</Badge>
              ) : (
                <Badge variant="secondary">Not Authenticated</Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isAuthenticated ? (
              <div className="space-y-4">
                <div className="p-4 bg-green-50 rounded-lg">
                  <p className="font-medium text-green-900">
                    Welcome, {currentUser?.name || 'User'}!
                  </p>
                  <p className="text-sm text-green-700">
                    Email: {currentUser?.email}
                  </p>
                </div>
                <Button onClick={handleLogout} variant="outline" size="sm">
                  <LogOut className="w-4 h-4 mr-2" />
                  Logout
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label>Name</Label>
                    <Input
                      value={authForm.name}
                      onChange={(e) => setAuthForm({...authForm, name: e.target.value})}
                    />
                  </div>
                  <div>
                    <Label>Email</Label>
                    <Input
                      type="email"
                      value={authForm.email}
                      onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
                    />
                  </div>
                  <div>
                    <Label>Password</Label>
                    <Input
                      type="password"
                      value={authForm.password}
                      onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
                    />
                  </div>
                </div>
                <div className="flex space-x-4">
                  <Button onClick={handleRegister} disabled={loading}>
                    {loading ? <Loader className="w-4 h-4 animate-spin mr-2" /> : <User className="w-4 h-4 mr-2" />}
                    Register
                  </Button>
                  <Button onClick={handleLogin} disabled={loading}>
                    {loading ? <Loader className="w-4 h-4 animate-spin mr-2" /> : <LogIn className="w-4 h-4 mr-2" />}
                    Login
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* AI Analysis Testing */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="w-5 h-5" />
              <span>AI Resume Analysis</span>
            </CardTitle>
            <CardDescription>
              Test the AI suggestions feature (works with or without authentication)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Target Role</Label>
              <Input
                value={targetRole}
                onChange={(e) => setTargetRole(e.target.value)}
                placeholder="e.g., Software Engineer"
              />
            </div>
            
            <div>
              <Label>Resume Text</Label>
              <textarea
                className="w-full h-48 p-3 border rounded-md"
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste your resume content here..."
              />
            </div>
            
            <Button 
              onClick={handleAIAnalysis} 
              disabled={loading || !resumeText.trim()}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin mr-2" />
                  Analyzing Resume...
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4 mr-2" />
                  Get AI Suggestions
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* AI Results Display */}
        {aiResults && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <span>AI Analysis Results</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Overall Score */}
                {aiResults.suggestions?.overall_score && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h3 className="font-semibold text-blue-900 mb-2">
                      Overall Score: {aiResults.suggestions.overall_score.grade} 
                      ({aiResults.suggestions.overall_score.percentage.toFixed(1)}%)
                    </h3>
                    <div className="space-y-1">
                      {aiResults.suggestions.overall_score.feedback.map((item: string, index: number) => (
                        <p key={index} className="text-sm text-blue-700">{item}</p>
                      ))}
                    </div>
                  </div>
                )}

                {/* Priority Actions */}
                {aiResults.suggestions?.priority_actions && (
                  <div>
                    <h3 className="font-semibold mb-3">🎯 Priority Actions</h3>
                    <div className="space-y-2">
                      {aiResults.suggestions.priority_actions.map((action: any, index: number) => (
                        <div key={index} className="p-3 bg-orange-50 rounded-lg">
                          <div className="flex items-start space-x-3">
                            <Badge className="bg-orange-100 text-orange-800">
                              #{action.priority}
                            </Badge>
                            <div>
                              <h4 className="font-medium text-orange-900">{action.title}</h4>
                              <p className="text-sm text-orange-700">{action.description}</p>
                              <div className="flex items-center space-x-2 mt-1">
                                <Badge variant="outline" className="text-xs">
                                  {action.impact} Impact
                                </Badge>
                                <Badge variant="secondary" className="text-xs">
                                  {action.time_estimate}
                                </Badge>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Skill Recommendations */}
                {aiResults.suggestions?.skill_recommendations && aiResults.suggestions.skill_recommendations.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-3">💪 Skill Recommendations</h3>
                    <div className="space-y-2">
                      {aiResults.suggestions.skill_recommendations.slice(0, 3).map((skill: any, index: number) => (
                        <div key={index} className="p-3 bg-purple-50 rounded-lg">
                          <h4 className="font-medium text-purple-900">{skill.title}</h4>
                          <p className="text-sm text-purple-700">{skill.description}</p>
                          <Badge className="mt-1 bg-purple-100 text-purple-800">
                            {skill.priority} priority
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Content Improvements */}
                {aiResults.suggestions?.content_improvements && aiResults.suggestions.content_improvements.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-3">📝 Content Improvements</h3>
                    <div className="space-y-2">
                      {aiResults.suggestions.content_improvements.slice(0, 2).map((improvement: any, index: number) => (
                        <div key={index} className="p-3 bg-green-50 rounded-lg">
                          <h4 className="font-medium text-green-900">{improvement.title}</h4>
                          <p className="text-sm text-green-700">{improvement.description}</p>
                          <p className="text-xs text-green-600 mt-1">💡 {improvement.action}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default TestPage;