"""
File processing utilities for resume and job description parsing.
"""

import io
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import os

# PDF processing
try:
    import fitz  # PyMuPDF
    import pdfplumber
except ImportError:
    fitz = None
    pdfplumber = None

# DOCX processing
try:
    from docx import Document
    import docx2txt
except ImportError:
    Document = None
    docx2txt = None

# NLP processing
try:
    import spacy
    from spacy.matcher import Matcher
except ImportError:
    spacy = None
    Matcher = None

from app.config import settings

logger = logging.getLogger(__name__)


class FileProcessor:
    """Base class for file processing."""
    
    def __init__(self):
        self.supported_extensions = settings.allowed_extensions
        self.max_file_size = settings.max_file_size
    
    def validate_file(self, file_content: bytes, file_name: str, file_size: int) -> bool:
        """Validate file before processing."""
        # Check file size
        if file_size > self.max_file_size:
            raise ValueError(f"File size {file_size} exceeds maximum allowed size {self.max_file_size}")
        
        # Check file extension
        file_extension = Path(file_name).suffix.lower().lstrip('.')
        if file_extension not in self.supported_extensions:
            raise ValueError(f"File extension '{file_extension}' not supported")
        
        # Check if file content is not empty
        if not file_content or len(file_content) == 0:
            raise ValueError("File content is empty")
        
        return True
    
    def extract_text(self, file_content: bytes, file_name: str) -> str:
        """Extract text from file based on its type."""
        file_extension = Path(file_name).suffix.lower()
        
        if file_extension == '.pdf':
            return self._extract_pdf_text(file_content)
        elif file_extension in ['.docx', '.doc']:
            return self._extract_docx_text(file_content)
        elif file_extension == '.txt':
            return self._extract_txt_text(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF file."""
        text = ""
        
        try:
            # Try PyMuPDF first
            if fitz:
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc[page_num]
                    text += page.get_text()
                pdf_doc.close()
            
            # Fallback to pdfplumber if PyMuPDF fails or produces poor results
            elif pdfplumber:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                raise ImportError("No PDF processing library available")
                
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return text.strip()
    
    def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX file."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            text = ""
            
            try:
                # Try docx2txt first (simpler and often more reliable)
                if docx2txt:
                    text = docx2txt.process(tmp_file_path)
                
                # Fallback to python-docx
                elif Document:
                    doc = Document(tmp_file_path)
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                else:
                    raise ImportError("No DOCX processing library available")
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"DOCX text extraction failed: {e}")
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the DOCX file")
        
        return text.strip()
    
    def _extract_txt_text(self, file_content: bytes) -> str:
        """Extract text from TXT file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    return text.strip()
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not decode text file with any supported encoding")
            
        except Exception as e:
            logger.error(f"TXT text extraction failed: {e}")
            raise ValueError(f"Failed to extract text from TXT: {str(e)}")


class ResumeProcessor(FileProcessor):
    """Specialized processor for resumes."""
    
    def __init__(self):
        super().__init__()
        self.nlp = self._load_nlp_model()
        self.skill_patterns = self._load_skill_patterns()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model."""
        if not spacy:
            logger.warning("spaCy not available, skill extraction will be limited")
            return None
        
        try:
            # Try to load English model
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            logger.warning("spaCy English model not found, skill extraction will be limited")
            return None
    
    def _load_skill_patterns(self) -> List[str]:
        """Load common skill patterns for extraction."""
        return [
            # Programming languages
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "php", "ruby",
            "swift", "kotlin", "scala", "r", "matlab", "sql", "html", "css",
            
            # Frameworks and libraries
            "react", "angular", "vue", "node.js", "express", "django", "flask", "spring", "laravel",
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "opencv",
            
            # Databases
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "oracle",
            
            # Cloud and DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
            "terraform", "ansible", "linux", "unix",
            
            # Data Science and AI
            "machine learning", "deep learning", "data science", "artificial intelligence",
            "natural language processing", "computer vision", "big data", "hadoop", "spark",
            
            # Web technologies
            "rest", "api", "graphql", "microservices", "websockets", "oauth", "jwt",
            
            # Testing
            "unit testing", "integration testing", "selenium", "jest", "pytest", "junit",
            
            # Project management
            "agile", "scrum", "kanban", "jira", "confluence", "slack", "trello"
        ]
    
    def process_resume(self, file_content: bytes, file_name: str) -> Dict:
        """Process resume and extract structured information."""
        # Validate and extract text
        self.validate_file(file_content, file_name, len(file_content))
        raw_text = self.extract_text(file_content, file_name)
        
        # Process text and extract information
        processed_data = {
            "raw_text": raw_text,
            "processed_text": self._clean_text(raw_text),
            "extracted_skills": self._extract_skills(raw_text),
            "extracted_experience": self._extract_experience(raw_text),
            "contact_info": self._extract_contact_info(raw_text),
            "education": self._extract_education(raw_text)
        }
        
        return processed_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s@.\-+()]+', ' ', text)
        return text.strip()
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text."""
        skills = []
        text_lower = text.lower()
        
        # Look for skill patterns
        for skill in self.skill_patterns:
            if skill.lower() in text_lower:
                skills.append(skill)
        
        # Use NLP if available for more sophisticated extraction
        if self.nlp:
            try:
                doc = self.nlp(text)
                # Extract entities that might be skills
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2:
                        skills.append(ent.text)
            except Exception as e:
                logger.warning(f"NLP skill extraction failed: {e}")
        
        # Remove duplicates and return
        return list(set(skills))
    
    def _extract_experience(self, text: str) -> Dict:
        """Extract work experience information."""
        experience = {
            "total_years": 0,
            "job_titles": [],
            "companies": [],
            "technologies": []
        }
        
        # Simple regex patterns for experience extraction
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        
        if years:
            years = [int(y) for y in years]
            experience["total_years"] = max(years) - min(years) if len(years) > 1 else 0
        
        # Extract job titles (basic patterns)
        job_patterns = [
            r'(?i)(software engineer|developer|programmer|analyst|manager|lead|senior|junior)',
            r'(?i)(data scientist|machine learning|ai engineer|devops|full stack)'
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text)
            experience["job_titles"].extend(matches)
        
        return experience
    
    def _extract_contact_info(self, text: str) -> Dict:
        """Extract contact information."""
        contact = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact["email"] = emails[0]
        
        # Phone pattern
        phone_pattern = r'[\+]?[1-9]?[0-9]{7,15}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact["phone"] = phones[0]
        
        return contact
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education information."""
        education = []
        
        # Education keywords
        edu_patterns = [
            r'(?i)(bachelor|master|phd|doctorate|degree|university|college)',
            r'(?i)(b\.?tech|m\.?tech|b\.?sc|m\.?sc|mba|bba)'
        ]
        
        for pattern in edu_patterns:
            matches = re.findall(pattern, text)
            education.extend(matches)
        
        return list(set(education))


class JobDescriptionProcessor(FileProcessor):
    """Specialized processor for job descriptions."""
    
    def __init__(self):
        super().__init__()
        self.nlp = self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model."""
        if not spacy:
            return None
        
        try:
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            return None
    
    def process_job_description(self, file_content: bytes, file_name: str, title: str) -> Dict:
        """Process job description and extract structured information."""
        # Extract text
        if file_content:
            self.validate_file(file_content, file_name, len(file_content))
            raw_text = self.extract_text(file_content, file_name)
        else:
            raw_text = ""
        
        # Process and extract information
        processed_data = {
            "raw_text": raw_text,
            "processed_text": self._clean_text(raw_text),
            "job_title": self._extract_job_title(raw_text, title),
            "must_have_skills": self._extract_required_skills(raw_text),
            "good_to_have_skills": self._extract_preferred_skills(raw_text),
            "qualifications": self._extract_qualifications(raw_text),
            "experience_required": self._extract_experience_requirements(raw_text),
            "location": self._extract_location(raw_text),
            "job_type": self._extract_job_type(raw_text)
        }
        
        return processed_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s@.\-+()]+', ' ', text)
        return text.strip()
    
    def _extract_job_title(self, text: str, provided_title: str) -> str:
        """Extract or use provided job title."""
        if provided_title:
            return provided_title
        
        # Try to extract from text
        title_patterns = [
            r'(?i)job title:?\s*([^\n\r]+)',
            r'(?i)position:?\s*([^\n\r]+)',
            r'(?i)role:?\s*([^\n\r]+)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return "Software Engineer"  # Default
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required/must-have skills."""
        skills = []
        
        # Look for required skills sections
        required_patterns = [
            r'(?i)required.{0,50}skills?:(.{1,500}?)(?:\n\n|\n[A-Z]|$)',
            r'(?i)must.{0,50}have:(.{1,500}?)(?:\n\n|\n[A-Z]|$)',
            r'(?i)essential.{0,50}skills?:(.{1,500}?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in required_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                skills.extend(self._extract_skills_from_text(match))
        
        return list(set(skills))
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred/good-to-have skills."""
        skills = []
        
        # Look for preferred skills sections
        preferred_patterns = [
            r'(?i)preferred.{0,50}skills?:(.{1,500}?)(?:\n\n|\n[A-Z]|$)',
            r'(?i)nice.{0,50}to.{0,50}have:(.{1,500}?)(?:\n\n|\n[A-Z]|$)',
            r'(?i)bonus.{0,50}skills?:(.{1,500}?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in preferred_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                skills.extend(self._extract_skills_from_text(match))
        
        return list(set(skills))
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from a text snippet."""
        skills = []
        
        # Common skill patterns
        skill_patterns = [
            "python", "java", "javascript", "react", "angular", "vue", "node.js",
            "docker", "kubernetes", "aws", "azure", "sql", "mongodb", "git"
        ]
        
        text_lower = text.lower()
        for skill in skill_patterns:
            if skill in text_lower:
                skills.append(skill)
        
        # Extract from bullet points or comma-separated lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # Extract skill from bullet point
                skill = re.sub(r'^[•\-*]\s*', '', line).strip()
                if skill and len(skill) < 50:  # Reasonable skill name length
                    skills.append(skill)
        
        return skills
    
    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract qualification requirements."""
        qualifications = []
        
        qual_patterns = [
            r'(?i)qualifications?:(.{1,500}?)(?:\n\n|\n[A-Z]|$)',
            r'(?i)requirements?:(.{1,500}?)(?:\n\n|\n[A-Z]|$)',
            r'(?i)education:(.{1,500}?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in qual_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Extract individual qualifications
                lines = match.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                        qual = re.sub(r'^[•\-*]\s*', '', line).strip()
                        if qual:
                            qualifications.append(qual)
        
        return qualifications
    
    def _extract_experience_requirements(self, text: str) -> str:
        """Extract experience requirements."""
        exp_patterns = [
            r'(?i)(\d+[\+]?\s*(?:to\s+\d+)?\s*years?\s*(?:of\s+)?experience)',
            r'(?i)(entry.level|junior|senior|lead|principal)',
            r'(?i)experience:?\s*([^\n\r]+)'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def _extract_location(self, text: str) -> str:
        """Extract job location."""
        location_patterns = [
            r'(?i)location:?\s*([^\n\r]+)',
            r'(?i)based in:?\s*([^\n\r]+)',
            r'(?i)office:?\s*([^\n\r]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Look for common city/state patterns
        city_pattern = r'\b(?:New York|San Francisco|Los Angeles|Chicago|Boston|Seattle|Austin|Denver|Remote)\b'
        match = re.search(city_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
        
        return "Not specified"
    
    def _extract_job_type(self, text: str) -> str:
        """Extract job type (full-time, part-time, etc.)."""
        type_patterns = [
            r'(?i)(full.time|part.time|contract|freelance|internship|temporary)',
            r'(?i)employment type:?\s*([^\n\r]+)'
        ]
        
        for pattern in type_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return "Full-time"  # Default


# Global processor instances
resume_processor = ResumeProcessor()
job_description_processor = JobDescriptionProcessor()


def get_resume_processor() -> ResumeProcessor:
    """Get the resume processor instance."""
    return resume_processor


def get_job_description_processor() -> JobDescriptionProcessor:
    """Get the job description processor instance."""
    return job_description_processor