/**
 * API Client for Resume Checker Backend
 * Handles all communication with the FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export interface ApiResponse<T = any> {
  status: 'success' | 'error';
  data?: T;
  message?: string;
  detail?: string;
}

export interface ResumeUploadResponse {
  id: string;
  file_name: string;
  file_url: string;
  processed_text: string;
  extracted_skills: string[];
  extracted_experience: Record<string, any>;
  uploaded_at: string;
}

export interface JobDescriptionResponse {
  id: string;
  title: string;
  company?: string;
  processed_text: string;
  must_have_skills: string[];
  good_to_have_skills: string[];
  qualifications: string[];
  experience_required: string;
  created_at: string;
}

export interface EvaluationResponse {
  id: string;
  relevance_score: number;
  verdict: 'High' | 'Medium' | 'Low';
  matched_skills: string[];
  missing_skills: string[];
  skill_match_percentage: number;
  experience_match: boolean;
  feedback: string;
  recommendations: string[];
  processing_time: number;
  created_at: string;
}

export interface AISuggestion {
  type: string;
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  action: string;
  impact: string;
}

export interface AISuggestionsResponse {
  status: 'success' | 'error';
  suggestions: {
    skill_recommendations: AISuggestion[];
    content_improvements: AISuggestion[];
    formatting_suggestions: AISuggestion[];
    keyword_optimization: AISuggestion[];
    ats_optimization: AISuggestion[];
    overall_score: {
      score: number;
      max_score: number;
      percentage: number;
      feedback: string[];
      grade: string;
    };
    priority_actions: Array<{
      priority: number;
      title: string;
      description: string;
      impact: string;
      time_estimate: string;
    }>;
  };
  target_role?: string;
  analyzed_at: string;
}

class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor() {
    this.baseUrl = API_BASE_URL;
    this.token = localStorage.getItem('auth_token');
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('auth_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...((options.headers as Record<string, string>) || {}),
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          status: 'error',
          message: data.detail || `HTTP ${response.status}`,
        };
      }

      return {
        status: 'success',
        data,
      };
    } catch (error) {
      return {
        status: 'error',
        message: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  private async uploadFile(
    endpoint: string,
    file: File,
    additionalData?: Record<string, string>
  ): Promise<ApiResponse> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const formData = new FormData();
    formData.append('file', file);
    
    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, value);
      });
    }

    const headers: Record<string, string> = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          status: 'error',
          message: data.detail || `HTTP ${response.status}`,
        };
      }

      return {
        status: 'success',
        data,
      };
    } catch (error) {
      return {
        status: 'error',
        message: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  // Authentication
  async login(email: string, password: string): Promise<ApiResponse<{ access_token: string; user: any }>> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async register(email: string, password: string, name: string): Promise<ApiResponse> {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, name }),
    });
  }

  async getCurrentUser(): Promise<ApiResponse> {
    return this.request('/auth/me');
  }

  // Resume Management
  async uploadResume(file: File): Promise<ApiResponse<ResumeUploadResponse>> {
    return this.uploadFile('/resumes/upload', file);
  }

  async getResumes(): Promise<ApiResponse<ResumeUploadResponse[]>> {
    return this.request('/resumes/');
  }

  async getResume(id: string): Promise<ApiResponse<ResumeUploadResponse>> {
    return this.request(`/resumes/${id}`);
  }

  async deleteResume(id: string): Promise<ApiResponse> {
    return this.request(`/resumes/${id}`, { method: 'DELETE' });
  }

  // Job Description Management
  async uploadJobDescription(file: File, title: string): Promise<ApiResponse<JobDescriptionResponse>> {
    return this.uploadFile('/job-descriptions/upload', file, { title });
  }

  async getJobDescriptions(): Promise<ApiResponse<JobDescriptionResponse[]>> {
    return this.request('/job-descriptions/');
  }

  async getJobDescription(id: string): Promise<ApiResponse<JobDescriptionResponse>> {
    return this.request(`/job-descriptions/${id}`);
  }

  // Evaluation
  async evaluateResume(resumeId: string, jdId: string): Promise<ApiResponse<EvaluationResponse>> {
    return this.request('/evaluations/evaluate', {
      method: 'POST',
      body: JSON.stringify({ resume_id: resumeId, jd_id: jdId }),
    });
  }

  async getEvaluations(): Promise<ApiResponse<EvaluationResponse[]>> {
    return this.request('/evaluations/');
  }

  async getEvaluation(id: string): Promise<ApiResponse<EvaluationResponse>> {
    return this.request(`/evaluations/${id}`);
  }

  // AI Suggestions
  async getAISuggestions(resumeText: string, targetRole?: string): Promise<ApiResponse<AISuggestionsResponse>> {
    return this.request('/api/v1/ai-suggestions', {
      method: 'POST',
      body: JSON.stringify({ 
        resume_text: resumeText, 
        target_role: targetRole 
      }),
    });
  }

  // Real-time AI Suggestions Streaming
  async streamAISuggestions(
    resumeText: string,
    targetRole: string | undefined,
    onUpdate: (data: any) => void,
    onError: (error: string) => void,
    onComplete: () => void
  ): Promise<void> {
    const url = `${this.baseUrl}/api/v1/ai-suggestions/stream`;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({ 
          resume_text: resumeText, 
          target_role: targetRole 
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          onComplete();
          break;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onUpdate(data);
            } catch (error) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Unknown error occurred');
    }
  }

  // Dashboard
  async getDashboardStats(): Promise<ApiResponse> {
    return this.request('/dashboard/stats');
  }

  async getRecentEvaluations(): Promise<ApiResponse> {
    return this.request('/dashboard/recent-evaluations');
  }

  // Health Check
  async healthCheck(): Promise<ApiResponse> {
    return this.request('/health');
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export types for use in components
export type {
  ResumeUploadResponse,
  JobDescriptionResponse,
  EvaluationResponse,
  AISuggestionsResponse,
  AISuggestion,
};