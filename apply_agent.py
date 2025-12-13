"""
Local AI Job Application Agent - Core Agent Logic
Uses browser-use + Playwright for DOM interaction and Ollama for reasoning.
"""

import asyncio
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

# Browser automation
from browser_use import Agent
from langchain_ollama import ChatOllama

# Local database for tracking applications
from application_db import get_db, ApplicationStatus

# Type definitions
LogCallback = Callable[[str, str], None]


@dataclass
class AgentResult:
    """Result of an agent run."""
    success: bool
    message: str
    application_id: Optional[int] = None
    fields_filled: int = 0
    research_queries: int = 0


def load_profile(profile_path: str) -> str:
    """Load the user profile from markdown file."""
    path = Path(profile_path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Profile not found at: {profile_path}")


def build_system_prompt(profile_content: str) -> str:
    """Build the system prompt for the agent with the user's profile data."""
    return f"""You are an autonomous job application agent. Your task is to fill out job application forms accurately and professionally using the candidate's profile data.

## CANDIDATE PROFILE (Use this data to fill forms)
{profile_content}

## INSTRUCTIONS

### Form Filling Rules:
1. **Match fields carefully**: Map form labels (e.g., "First Name", "Email", "Phone") to the profile data above.
2. **Be precise**: Use exact values from the profile. Don't paraphrase names, emails, or phone numbers.
3. **Handle dropdowns**: Select the closest matching option for dropdown fields.
4. **File uploads**: If asked to upload a resume, look for an upload button and use the file selector.

### Research Protocol (Multi-Tab):
When you encounter questions like "Why do you want to work here?" or "What interests you about this role?":
1. **Open a new tab** (don't close the application tab).
2. **Search Google** for: "[Company Name] mission values about us"
3. **Read** the company's About page or mission statement.
4. **Synthesize** an answer that connects the candidate's skills/interests to the company's values.
5. **Close the research tab** and return to the application.
6. **Fill in** the synthesized answer.

### Salary Questions:
If asked about salary expectations:
1. Open a new tab and search for average salary for the role + location.
2. Use a reasonable market rate based on the research.
3. If unsure, use a range format (e.g., "$80,000 - $100,000").

### Critical Safety Rules:
1. **NEVER submit** the application without explicit user confirmation.
2. **STOP and alert** if you detect a CAPTCHA - do not attempt to solve it.
3. **STOP and alert** if you need to log in - user must handle authentication.
4. **STOP and alert** if you're unsure about a field - better to ask than guess wrong.
5. **Be patient**: Wait for pages to load fully before interacting.

### Navigation Tips:
- Look for "Apply", "Submit Application", or similar buttons to start.
- Handle multi-page applications by clicking "Next" or "Continue".
- Watch for confirmation messages to know when you've completed sections.

You are applying as: {profile_content.split('Name:')[1].split('\n')[0].strip() if 'Name:' in profile_content else 'the candidate'}
"""


async def run_agent(
    job_url: str,
    resume_path: str,
    profile_path: str,
    model_name: str = "qwen2.5:7b",
    log_callback: Optional[LogCallback] = None,
    company: Optional[str] = None,
    role: Optional[str] = None,
    skip_duplicate_check: bool = False
) -> dict:
    """
    Run the job application agent.
    
    Args:
        job_url: The URL of the job posting/application page.
        resume_path: Path to the resume PDF file.
        profile_path: Path to the my_profile.md file.
        model_name: The Ollama model to use (default: llama3.1).
        log_callback: Optional callback function for logging (message, level).
        company: Company name (optional, for tracking).
        role: Role title (optional, for tracking).
        skip_duplicate_check: If True, skip duplicate URL check.
    
    Returns:
        dict with 'success' (bool), 'message' (str), and 'application_id' (int).
    """
    
    def log(message: str, level: str = "info"):
        """Log a message using the callback if provided."""
        if log_callback:
            log_callback(message, level)
        print(f"[{level.upper()}] {message}")
    
    db = get_db()
    application_id = None
    
    try:
        # Check for duplicate application
        if not skip_duplicate_check:
            existing = db.get_application_by_url(job_url)
            if existing:
                log(f"⚠️ Already applied to this job on {existing.created_at.strftime('%Y-%m-%d')}", "warning")
                return {
                    "success": False,
                    "message": f"Duplicate application: Already applied to {existing.company} - {existing.role}",
                    "application_id": existing.id,
                    "is_duplicate": True
                }
        
        # Record application in database
        application_id = db.add_application(
            company=company or "Unknown Company",
            role=role or "Unknown Role",
            job_url=job_url,
            status=ApplicationStatus.IN_PROGRESS,
            resume_used=resume_path
        )
        
        if application_id:
            log(f"📝 Application #{application_id} recorded in database", "info")
        
        # Load profile
        log("Loading candidate profile...", "info")
        profile_content = load_profile(profile_path)
        log(f"Profile loaded ({len(profile_content)} characters)", "success")
        
        # Build system prompt
        system_prompt = build_system_prompt(profile_content)
        
        # Initialize the LLM
        log(f"Initializing Ollama with model: {model_name}", "info")
        llm = ChatOllama(
            model=model_name,
            temperature=0.1,  # Low temperature for consistent form filling
            num_ctx=8192,     # Context window for the profile + page content
        )
        log("LLM initialized successfully", "success")
        
        # Initialize the browser agent
        log("Launching browser...", "action")
        
        # Create the task for the agent
        task = f"""
NAVIGATE TO: {job_url}

YOUR GOAL: 
Complete the job application form up to the review stage. DO NOT SUBMIT.

CANDIDATE PROFILE (Source of Truth):
{profile_content}

EXECUTION STEPS:
1. **Login/Start:** If a login is required, look for "Apply without account" or "Apply with Resume". If blocked by a complex login, PAUSE and ask user.
2. **Resume Upload:** Upload the file at: "{resume_path}".
3. **Form Filling:** - Map profile data to fields (e.g., "Experience" -> "Reynolds and Reynolds").
   - If a field asks for "Desired Salary" and it's not in the profile, OPEN A NEW TAB, search "average salary for [Role] at [Company]", extract the number, close tab, and fill it.
   - If a field asks "Why do you want to work here?", OPEN A NEW TAB, read the company's "About Us", and synthesize a short answer connecting their values to my "Thematic Interests" (in profile).
4. **Review:** Click "Next" until you reach the "Review" or "Final Submit" page.
5. **HALT:** Stop immediately upon reaching the final page. notify the user: "Application ready for review."

CRITICAL RULES:
- **NO SUBMITTING:** Never click "Submit Application".
- **CAPTCHAS:** If you see a CAPTCHA, stop and alert the user.
- **PRIVACY:** Do not hallucinate personal data. Use *only* what is in the profile.
"""
        
        # Initialize and run the browser-use agent
        # Note: Type ignore needed due to langchain type stub version mismatch
        agent = Agent(
            task=task,
            llm=llm,  # type: ignore[arg-type]
            extend_system_message=system_prompt,  # Inject candidate profile + instructions
            browser_context_kwargs={
                "headless": False,  # Show the browser so user can watch/intervene
            }
        )
        
        log(f"Navigating to: {job_url}", "action")
        
        # Run the agent
        result = await agent.run()
        
        log("Agent completed its run", "success")
        
        # Update database status to completed
        if application_id:
            db.update_status(application_id, ApplicationStatus.COMPLETED)
            log(f"✅ Application #{application_id} marked as completed", "success")
        
        return {
            "success": True,
            "message": "Agent completed. Please review the application before submitting.",
            "application_id": application_id,
            "result": result
        }
        
    except FileNotFoundError as e:
        log(f"Profile file error: {e}", "error")
        if application_id:
            db.update_status(application_id, ApplicationStatus.FAILED, notes=str(e))
        return {"success": False, "message": str(e), "application_id": application_id}
    
    except ConnectionError:
        log("Could not connect to Ollama. Is it running?", "error")
        if application_id:
            db.update_status(application_id, ApplicationStatus.FAILED, notes="Ollama connection failed")
        return {
            "success": False, 
            "message": "Ollama connection failed. Please ensure Ollama is running (ollama serve).",
            "application_id": application_id
        }
    
    except Exception as e:
        log(f"Unexpected error: {type(e).__name__}: {e}", "error")
        if application_id:
            db.update_status(application_id, ApplicationStatus.FAILED, notes=f"{type(e).__name__}: {e}")
        return {"success": False, "message": f"Agent error: {str(e)}", "application_id": application_id}


# For testing the agent directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python apply_agent.py <job_url> <resume_path>")
        print("Example: python apply_agent.py https://boards.greenhouse.io/company/jobs/123 /path/to/resume.pdf")
        sys.exit(1)
    
    job_url = sys.argv[1]
    resume_path = sys.argv[2]
    profile_path = Path(__file__).parent / "my_profile.md"
    
    print(f"Starting agent for: {job_url}")
    print(f"Using profile: {profile_path}")
    print(f"Using resume: {resume_path}")
    
    result = asyncio.run(run_agent(
        job_url=job_url,
        resume_path=resume_path,
        profile_path=str(profile_path),
        model_name="qwen2.5:7b"
    ))
    
    print(f"\nResult: {result}")