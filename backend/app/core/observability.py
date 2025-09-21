import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
    from langsmith.schemas import Run, Example
except ImportError:
    Client = None
    traceable = None

from app.config import settings

logger = logging.getLogger(__name__)


class LangSmithManager:
    """LangSmith client manager for observability."""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.enabled = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LangSmith client if available."""
        try:
            if settings.langchain_api_key and Client:
                # Set environment variables for LangSmith
                os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
                os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
                os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
                os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
                
                self.client = Client(
                    api_url=settings.langchain_endpoint,
                    api_key=settings.langchain_api_key
                )
                
                self.enabled = True
                logger.info("LangSmith client initialized successfully")
                
            else:
                logger.warning("LangSmith not available - missing API key or package")
                self.enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self.enabled = False
    
    def log_evaluation_run(
        self,
        evaluation_id: str,
        resume_id: str,
        jd_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log an evaluation run to LangSmith."""
        if not self.enabled:
            return
        
        try:
            run_data = {
                "name": "resume_evaluation",
                "run_type": "chain",
                "inputs": inputs,
                "outputs": outputs,
                "start_time": datetime.utcnow(),
                "end_time": datetime.utcnow(),
                "extra": {
                    "evaluation_id": evaluation_id,
                    "resume_id": resume_id,
                    "jd_id": jd_id,
                    "metadata": metadata or {}
                }
            }
            
            if error:
                run_data["error"] = error
            
            # Create run using client
            self.client.create_run(**run_data)
            
            logger.info(f"Logged evaluation run to LangSmith: {evaluation_id}")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation run to LangSmith: {e}")
    
    def log_llm_call(
        self,
        evaluation_id: str,
        model_name: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log an LLM call to LangSmith."""
        if not self.enabled:
            return
        
        try:
            run_data = {
                "name": f"llm_call_{model_name}",
                "run_type": "llm",
                "inputs": {"prompt": prompt},
                "outputs": {"response": response},
                "start_time": datetime.utcnow(),
                "end_time": datetime.utcnow(),
                "extra": {
                    "evaluation_id": evaluation_id,
                    "model_name": model_name,
                    "metadata": metadata or {}
                }
            }
            
            if error:
                run_data["error"] = error
            
            self.client.create_run(**run_data)
            
            logger.info(f"Logged LLM call to LangSmith: {evaluation_id}")
            
        except Exception as e:
            logger.error(f"Failed to log LLM call to LangSmith: {e}")
    
    def create_evaluation_dataset(
        self,
        dataset_name: str,
        examples: list[Dict[str, Any]]
    ):
        """Create an evaluation dataset in LangSmith."""
        if not self.enabled:
            return None
        
        try:
            # Create dataset
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Resume evaluation dataset"
            )
            
            # Add examples
            for example in examples:
                self.client.create_example(
                    dataset_id=dataset.id,
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                    metadata=example.get("metadata", {})
                )
            
            logger.info(f"Created evaluation dataset: {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create evaluation dataset: {e}")
            return None
    
    def get_run_stats(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about runs in LangSmith."""
        if not self.enabled:
            return {}
        
        try:
            project = project_name or settings.langchain_project
            
            # Get runs for the project
            runs = list(self.client.list_runs(project_name=project, limit=100))
            
            if not runs:
                return {"total_runs": 0}
            
            # Calculate stats
            total_runs = len(runs)
            successful_runs = len([r for r in runs if not r.error])
            failed_runs = total_runs - successful_runs
            
            # Calculate average duration
            durations = []
            for run in runs:
                if run.start_time and run.end_time:
                    duration = (run.end_time - run.start_time).total_seconds()
                    durations.append(duration)
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            return {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "average_duration_seconds": avg_duration
            }
            
        except Exception as e:
            logger.error(f"Failed to get run stats: {e}")
            return {}


class MetricsCollector:
    """Collect and store application metrics."""
    
    def __init__(self):
        self.metrics = {
            "evaluations": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "average_processing_time": 0.0
            },
            "uploads": {
                "resumes": 0,
                "job_descriptions": 0,
                "total_file_size": 0
            },
            "users": {
                "total_students": 0,
                "total_placement_teams": 0,
                "active_users": 0
            }
        }
    
    def record_evaluation(
        self,
        evaluation_id: str,
        processing_time: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record an evaluation metric."""
        self.metrics["evaluations"]["total"] += 1
        
        if success:
            self.metrics["evaluations"]["successful"] += 1
        else:
            self.metrics["evaluations"]["failed"] += 1
        
        # Update average processing time
        total_evals = self.metrics["evaluations"]["total"]
        current_avg = self.metrics["evaluations"]["average_processing_time"]
        new_avg = ((current_avg * (total_evals - 1)) + processing_time) / total_evals
        self.metrics["evaluations"]["average_processing_time"] = new_avg
        
        logger.info(f"Recorded evaluation metric: {evaluation_id}, time: {processing_time:.2f}s, success: {success}")
    
    def record_upload(
        self,
        file_type: str,  # 'resume' or 'job_description'
        file_size: int,
        success: bool
    ):
        """Record a file upload metric."""
        if success:
            if file_type == "resume":
                self.metrics["uploads"]["resumes"] += 1
            elif file_type == "job_description":
                self.metrics["uploads"]["job_descriptions"] += 1
            
            self.metrics["uploads"]["total_file_size"] += file_size
        
        logger.info(f"Recorded upload metric: {file_type}, size: {file_size}, success: {success}")
    
    def record_user_activity(self, user_role: str):
        """Record user activity."""
        if user_role == "student":
            self.metrics["users"]["total_students"] += 1
        elif user_role == "placement_team":
            self.metrics["users"]["total_placement_teams"] += 1
        
        self.metrics["users"]["active_users"] += 1
        
        logger.info(f"Recorded user activity: {user_role}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        for category in self.metrics:
            for metric in self.metrics[category]:
                if isinstance(self.metrics[category][metric], (int, float)):
                    self.metrics[category][metric] = 0


class PerformanceMonitor:
    """Monitor application performance."""
    
    def __init__(self):
        self.slow_requests = []
        self.error_counts = {}
    
    def record_slow_request(
        self,
        endpoint: str,
        duration: float,
        threshold: float = 5.0
    ):
        """Record slow requests."""
        if duration > threshold:
            self.slow_requests.append({
                "endpoint": endpoint,
                "duration": duration,
                "timestamp": datetime.utcnow()
            })
            
            # Keep only last 100 slow requests
            if len(self.slow_requests) > 100:
                self.slow_requests = self.slow_requests[-100:]
            
            logger.warning(f"Slow request detected: {endpoint} took {duration:.2f}s")
    
    def record_error(self, error_type: str, endpoint: str):
        """Record application errors."""
        key = f"{error_type}:{endpoint}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        logger.error(f"Error recorded: {error_type} at {endpoint}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return {
            "slow_requests": self.slow_requests[-10:],  # Last 10 slow requests
            "error_counts": self.error_counts,
            "total_slow_requests": len(self.slow_requests)
        }


# Global instances
langsmith_manager = LangSmithManager()
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor()


def get_langsmith_manager() -> LangSmithManager:
    """Get LangSmith manager instance."""
    return langsmith_manager


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance."""
    return metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance."""
    return performance_monitor


# Decorator for tracing evaluation functions
def trace_evaluation(func):
    """Decorator to trace evaluation functions with LangSmith."""
    if not langsmith_manager.enabled or not traceable:
        return func
    
    return traceable(
        name=func.__name__,
        project_name=settings.langchain_project
    )(func)