import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class MockAIServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        if path == '/' or path == '/health':
            self._handle_health_check()
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {"status": "error", "message": "Endpoint not found"}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def do_POST(self):
        """Handle POST requests"""
        path = urlparse(self.path).path
        
        if path == '/api/v1/ai-suggestions':
            self._handle_ai_suggestions()
        elif path == '/api/v1/ai-suggestions/stream':
            self._handle_ai_suggestions_stream()
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {"status": "error", "message": "Endpoint not found"}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def _set_cors_headers(self):
        """Set CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def _handle_health_check(self):
        """Handle health check requests"""
        try:
            self.send_response(200)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = {
                "status": "success",
                "message": "Mock AI Server is running",
                "timestamp": time.time(),
                "endpoints": [
                    "GET /health - Health check",
                    "POST /api/v1/ai-suggestions - Regular AI suggestions",
                    "POST /api/v1/ai-suggestions/stream - Streaming AI suggestions"
                ]
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def _handle_ai_suggestions(self):
        """Handle regular AI suggestions request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Simulate AI processing delay
            time.sleep(2)
            
            # Mock response
            response = {
                "status": "success",
                "message": "AI suggestions generated successfully",
                "data": {
                    "suggestions": {
                        "overall_score": {
                            "score": 78,
                            "max_score": 100,
                            "percentage": 78.0,
                            "grade": "B+",
                            "feedback": [
                                "Strong technical skills demonstrated",
                                "Good project experience shown",
                                "Consider adding more quantified achievements"
                            ]
                        },
                        "priority_actions": [
                            {
                                "priority": 1,
                                "title": "Add Quantified Achievements",
                                "description": "Include specific numbers and metrics in your accomplishments",
                                "impact": "High",
                                "time_estimate": "2-3 hours"
                            },
                            {
                                "priority": 2,
                                "title": "Update Technical Skills",
                                "description": "Add trending technologies like Docker and Kubernetes",
                                "impact": "Medium",
                                "time_estimate": "1 hour"
                            }
                        ],
                        "skill_recommendations": [
                            {
                                "type": "skill_recommendations",
                                "title": "Add Docker & Containerization",
                                "description": "Docker is highly sought after for modern software development roles",
                                "action": "Include Docker, Kubernetes, and container orchestration in your skills section",
                                "priority": "high",
                                "impact": "Increases job match by 25%"
                            },
                            {
                                "type": "skill_recommendations",
                                "title": "Cloud Platforms",
                                "description": "Cloud experience is essential for modern development",
                                "action": "Add AWS, Azure, or Google Cloud Platform experience",
                                "priority": "medium",
                                "impact": "Expands job opportunities"
                            }
                        ],
                        "content_improvements": [
                            {
                                "type": "content_improvements",
                                "title": "Quantify Your Impact",
                                "description": "Add specific metrics to demonstrate your achievements",
                                "action": "Replace vague statements with quantified results (e.g., 'Improved performance by 40%')",
                                "priority": "high",
                                "impact": "Makes achievements more compelling"
                            },
                            {
                                "type": "content_improvements",
                                "title": "Add Leadership Examples",
                                "description": "Include examples of leadership and collaboration",
                                "action": "Describe team leadership, mentoring, or cross-functional collaboration experiences",
                                "priority": "medium",
                                "impact": "Shows soft skills and growth potential"
                            }
                        ],
                        "formatting_suggestions": [
                            {
                                "type": "formatting_suggestions",
                                "title": "Use Action Verbs",
                                "description": "Start bullet points with strong action verbs",
                                "action": "Begin each accomplishment with verbs like 'Developed', 'Implemented', 'Led', 'Optimized'",
                                "priority": "low",
                                "impact": "Makes content more engaging"
                            }
                        ],
                        "keyword_optimization": [
                            {
                                "type": "keyword_optimization",
                                "title": "Add Industry Keywords",
                                "description": "Include trending software engineering keywords",
                                "action": "Add terms like 'microservices', 'API development', 'agile methodologies'",
                                "priority": "medium",
                                "impact": "Improves ATS compatibility"
                            }
                        ],
                        "ats_optimization": [
                            {
                                "type": "ats_optimization",
                                "title": "Use Standard Section Headers",
                                "description": "Ensure ATS can parse your resume sections",
                                "action": "Use standard headers like 'EXPERIENCE', 'EDUCATION', 'SKILLS'",
                                "priority": "low",
                                "impact": "Improves ATS parsing accuracy"
                            }
                        ]
                    }
                }
            }
            
            self.send_response(200)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self._set_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {
                "status": "error",
                "message": f"Server error: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def _handle_ai_suggestions_stream(self):
        """Handle streaming AI suggestions request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            self.send_response(200)
            self._set_cors_headers()
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            # Stream simulation
            events = [
                {"type": "status", "message": "Starting AI analysis..."},
                {"type": "status", "message": "Extracting skills from resume..."},
                {"type": "skills_extracted", "skills": ["JavaScript", "React", "Node.js", "Python", "MongoDB"]},
                {"type": "status", "message": "Generating skill recommendations..."},
                {
                    "type": "skill_recommendations",
                    "suggestions": [
                        {
                            "type": "skill_recommendations",
                            "title": "Add Docker & Containerization",
                            "description": "Docker is highly sought after for modern software development roles",
                            "action": "Include Docker, Kubernetes, and container orchestration in your skills section",
                            "priority": "high",
                            "impact": "Increases job match by 25%"
                        }
                    ]
                },
                {"type": "status", "message": "Analyzing content improvements..."},
                {
                    "type": "content_improvements",
                    "suggestions": [
                        {
                            "type": "content_improvements",
                            "title": "Quantify Your Impact",
                            "description": "Add specific metrics to demonstrate your achievements",
                            "action": "Replace vague statements with quantified results",
                            "priority": "high",
                            "impact": "Makes achievements more compelling"
                        }
                    ]
                },
                {"type": "status", "message": "Finalizing recommendations..."},
                {"type": "complete", "message": "Analysis complete!"}
            ]
            
            for event in events:
                event_data = f"data: {json.dumps(event)}\n\n"
                self.wfile.write(event_data.encode('utf-8'))
                self.wfile.flush()
                time.sleep(1)  # Simulate processing time
                
        except Exception as e:
            error_event = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            self.wfile.write(error_event.encode('utf-8'))

def start_mock_server():
    server = HTTPServer(('localhost', 8000), MockAIServer)
    print("🚀 Mock AI Server running on http://localhost:8000")
    print("📝 Available endpoints:")
    print("   POST /api/v1/ai-suggestions - Regular AI suggestions")
    print("   POST /api/v1/ai-suggestions/stream - Streaming AI suggestions")
    print("💡 Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⭐ Mock server stopped")
        server.shutdown()

if __name__ == "__main__":
    start_mock_server()