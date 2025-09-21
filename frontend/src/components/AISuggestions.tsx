import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Lightbulb, 
  TrendingUp, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  Sparkles,
  Target,
  FileText,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { apiClient, AISuggestion, AISuggestionsResponse } from '@/lib/api';

interface AISuggestionsProps {
  resumeText: string;
  targetRole?: string;
  onSuggestionsGenerated?: (suggestions: AISuggestionsResponse) => void;
}

interface StreamingData {
  type: string;
  message?: string;
  skills?: string[];
  suggestions?: AISuggestion[];
}

const AISuggestionsComponent: React.FC<AISuggestionsProps> = ({
  resumeText,
  targetRole,
  onSuggestionsGenerated,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('');
  const [extractedSkills, setExtractedSkills] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<AISuggestionsResponse | null>(null);
  const [streamingSuggestions, setStreamingSuggestions] = useState<{
    skill_recommendations: AISuggestion[];
    content_improvements: AISuggestion[];
    formatting_suggestions: AISuggestion[];
    keyword_optimization: AISuggestion[];
    ats_optimization: AISuggestion[];
  }>({
    skill_recommendations: [],
    content_improvements: [],
    formatting_suggestions: [],
    keyword_optimization: [],
    ats_optimization: [],
  });
  const [error, setError] = useState<string | null>(null);

  const generateSuggestions = async () => {
    if (!resumeText.trim()) {
      setError('Please provide resume content to analyze');
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuggestions(null);

    try {
      const response = await apiClient.getAISuggestions(resumeText, targetRole);
      
      if (response.status === 'success' && response.data) {
        setSuggestions(response.data);
        onSuggestionsGenerated?.(response.data);
      } else {
        setError(response.message || 'Failed to generate suggestions');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const startStreamingSuggestions = async () => {
    if (!resumeText.trim()) {
      setError('Please provide resume content to analyze');
      return;
    }

    setIsStreaming(true);
    setError(null);
    setCurrentStatus('Initializing AI analysis...');
    setExtractedSkills([]);
    setStreamingSuggestions({
      skill_recommendations: [],
      content_improvements: [],
      formatting_suggestions: [],
      keyword_optimization: [],
      ats_optimization: [],
    });

    await apiClient.streamAISuggestions(
      resumeText,
      targetRole,
      (data: StreamingData) => {
        switch (data.type) {
          case 'status':
            setCurrentStatus(data.message || '');
            break;
          case 'skills_extracted':
            setExtractedSkills(data.skills || []);
            break;
          case 'skill_recommendations':
          case 'content_improvements':
          case 'formatting_suggestions':
          case 'keyword_optimization':
          case 'ats_optimization':
            setStreamingSuggestions(prev => ({
              ...prev,
              [data.type]: data.suggestions || [],
            }));
            break;
          case 'complete':
            setCurrentStatus('Analysis complete!');
            setIsStreaming(false);
            break;
          case 'error':
            setError(data.message || 'An error occurred during streaming');
            setIsStreaming(false);
            break;
        }
      },
      (error: string) => {
        setError(error);
        setIsStreaming(false);
      },
      () => {
        setIsStreaming(false);
      }
    );
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'high':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'medium':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'low':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      default:
        return <Lightbulb className="w-4 h-4 text-blue-500" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950';
      case 'medium':
        return 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950';
      case 'low':
        return 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950';
      default:
        return 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950';
    }
  };

  const renderSuggestionCard = (suggestion: AISuggestion, index: number) => (
    <motion.div
      key={`${suggestion.type}-${index}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className={`p-4 rounded-lg border ${getPriorityColor(suggestion.priority)} hover:shadow-md transition-all duration-200`}
    >
      <div className="flex items-start space-x-3">
        {getPriorityIcon(suggestion.priority)}
        <div className="flex-1">
          <h4 className="font-semibold text-sm text-foreground mb-1">
            {suggestion.title}
          </h4>
          <p className="text-xs text-muted-foreground mb-2">
            {suggestion.description}
          </p>
          <p className="text-xs text-foreground mb-2 font-medium">
            💡 {suggestion.action}
          </p>
          <div className="flex items-center justify-between">
            <Badge variant="outline" className="text-xs">
              {suggestion.impact}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {suggestion.priority} priority
            </Badge>
          </div>
        </div>
      </div>
    </motion.div>
  );

  const renderSuggestionsSection = (
    title: string,
    icon: React.ReactNode,
    suggestions: AISuggestion[],
    description: string
  ) => {
    if (suggestions.length === 0) return null;

    return (
      <Card className="mb-6">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center space-x-2 text-lg">
            {icon}
            <span>{title}</span>
            <Badge variant="secondary">{suggestions.length}</Badge>
          </CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {suggestions.map((suggestion, index) => renderSuggestionCard(suggestion, index))}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-foreground">AI Resume Analysis</h2>
            <p className="text-muted-foreground">
              Get personalized suggestions to improve your resume{targetRole && ` for ${targetRole} roles`}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4">
          <Button
            onClick={generateSuggestions}
            disabled={isLoading || isStreaming}
            className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Generate Suggestions
              </>
            )}
          </Button>

          <Button
            onClick={startStreamingSuggestions}
            disabled={isLoading || isStreaming}
            variant="outline"
            className="border-purple-200 hover:bg-purple-50"
          >
            {isStreaming ? (
              <>
                <div className="w-4 h-4 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mr-2" />
                Live Analysis...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Real-time Analysis
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert className="mb-6 border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Streaming Status */}
      <AnimatePresence>
        {isStreaming && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6"
          >
            <Card className="border-purple-200 bg-purple-50 dark:bg-purple-950/20">
              <CardContent className="p-4">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
                  <div className="flex-1">
                    <p className="font-medium text-purple-900 dark:text-purple-100">
                      {currentStatus}
                    </p>
                    {extractedSkills.length > 0 && (
                      <div className="mt-2">
                        <p className="text-sm text-purple-700 dark:text-purple-300 mb-1">
                          Extracted Skills:
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {extractedSkills.slice(0, 10).map((skill, index) => (
                            <Badge key={index} variant="secondary" className="text-xs">
                              {skill}
                            </Badge>
                          ))}
                          {extractedSkills.length > 10 && (
                            <Badge variant="outline" className="text-xs">
                              +{extractedSkills.length - 10} more
                            </Badge>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Regular Suggestions Display */}
      {suggestions && !isStreaming && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          {/* Overall Score */}
          {suggestions.suggestions.overall_score && (
            <Card className="mb-6 border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950/20">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="w-5 h-5 text-green-600" />
                  <span>Overall Resume Score</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-4 mb-4">
                  <div className="text-3xl font-bold text-green-600">
                    {suggestions.suggestions.overall_score.grade}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-sm font-medium">
                        {suggestions.suggestions.overall_score.score}/{suggestions.suggestions.overall_score.max_score}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        ({suggestions.suggestions.overall_score.percentage.toFixed(1)}%)
                      </span>
                    </div>
                    <Progress 
                      value={suggestions.suggestions.overall_score.percentage} 
                      className="h-2"
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  {suggestions.suggestions.overall_score.feedback.map((item, index) => (
                    <p key={index} className="text-sm text-muted-foreground">
                      {item}
                    </p>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Priority Actions */}
          {suggestions.suggestions.priority_actions && suggestions.suggestions.priority_actions.length > 0 && (
            <Card className="mb-6 border-orange-200 bg-orange-50 dark:border-orange-800 dark:bg-orange-950/20">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5 text-orange-600" />
                  <span>Top Priority Actions</span>
                </CardTitle>
                <CardDescription>
                  Focus on these high-impact improvements first
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {suggestions.suggestions.priority_actions.map((action, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 bg-white dark:bg-gray-800 rounded-lg border">
                      <div className="w-6 h-6 bg-orange-100 dark:bg-orange-900 text-orange-600 rounded-full flex items-center justify-center text-sm font-bold">
                        {action.priority}
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-sm mb-1">{action.title}</h4>
                        <p className="text-xs text-muted-foreground mb-2">{action.description}</p>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">{action.impact}</Badge>
                          <Badge variant="secondary" className="text-xs">{action.time_estimate}</Badge>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Suggestion Categories */}
          {renderSuggestionsSection(
            "Skill Recommendations",
            <TrendingUp className="w-5 h-5 text-blue-600" />,
            suggestions.suggestions.skill_recommendations,
            "Trending skills and technologies to add to your resume"
          )}

          {renderSuggestionsSection(
            "Content Improvements",
            <FileText className="w-5 h-5 text-green-600" />,
            suggestions.suggestions.content_improvements,
            "Ways to enhance your resume content and impact"
          )}

          {renderSuggestionsSection(
            "Formatting Suggestions",
            <Lightbulb className="w-5 h-5 text-purple-600" />,
            suggestions.suggestions.formatting_suggestions,
            "Improve the structure and layout of your resume"
          )}

          {renderSuggestionsSection(
            "Keyword Optimization",
            <Target className="w-5 h-5 text-pink-600" />,
            suggestions.suggestions.keyword_optimization,
            "Industry-specific keywords to improve discoverability"
          )}

          {renderSuggestionsSection(
            "ATS Optimization",
            <CheckCircle className="w-5 h-5 text-indigo-600" />,
            suggestions.suggestions.ats_optimization,
            "Make your resume more compatible with Applicant Tracking Systems"
          )}
        </motion.div>
      )}

      {/* Streaming Suggestions Display */}
      {isStreaming && (
        <div>
          {renderSuggestionsSection(
            "Skill Recommendations",
            <TrendingUp className="w-5 h-5 text-blue-600" />,
            streamingSuggestions.skill_recommendations,
            "Trending skills and technologies to add to your resume"
          )}

          {renderSuggestionsSection(
            "Content Improvements",
            <FileText className="w-5 h-5 text-green-600" />,
            streamingSuggestions.content_improvements,
            "Ways to enhance your resume content and impact"
          )}

          {renderSuggestionsSection(
            "Formatting Suggestions",
            <Lightbulb className="w-5 h-5 text-purple-600" />,
            streamingSuggestions.formatting_suggestions,
            "Improve the structure and layout of your resume"
          )}

          {renderSuggestionsSection(
            "Keyword Optimization",
            <Target className="w-5 h-5 text-pink-600" />,
            streamingSuggestions.keyword_optimization,
            "Industry-specific keywords to improve discoverability"
          )}

          {renderSuggestionsSection(
            "ATS Optimization",
            <CheckCircle className="w-5 h-5 text-indigo-600" />,
            streamingSuggestions.ats_optimization,
            "Make your resume more compatible with Applicant Tracking Systems"
          )}
        </div>
      )}
    </div>
  );
};

export default AISuggestionsComponent;