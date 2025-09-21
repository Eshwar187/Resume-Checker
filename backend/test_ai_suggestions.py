import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.api.v1.ai_suggestions import ai_suggestions_router
from app.core.evaluation import ResumeEvaluator

async def test_ai_suggestions():
    """Test the AI suggestions functionality"""
    
    # Sample resume text
    resume_text = """
    John Doe
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
    - Deployed on AWS with CI/CD pipeline
    """
    
    target_role = "Software Engineer"
    
    try:
        # Initialize evaluator
        evaluator = ResumeEvaluator()
        
        # Test AI suggestions generation
        print("Testing AI suggestions generation...")
        suggestions = evaluator.generate_ai_suggestions(resume_text, target_role)
        
        print(f"✅ AI suggestions generated successfully!")
        print(f"Suggestions keys: {list(suggestions.keys())}")
        
        if 'skill_recommendations' in suggestions:
            print(f"Skill recommendations: {len(suggestions['skill_recommendations'])} items")
        
        if 'content_improvements' in suggestions:
            print(f"Content improvements: {len(suggestions['content_improvements'])} items")
        
        if 'overall_score' in suggestions:
            print(f"Overall score: {suggestions['overall_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI suggestions: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_ai_suggestions())
    if result:
        print("\n🎉 AI suggestions test completed successfully!")
    else:
        print("\n💥 AI suggestions test failed!")