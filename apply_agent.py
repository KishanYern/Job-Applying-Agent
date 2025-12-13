"""
Local AI Job Application Agent - Core Agent Logic
Uses browser-use + Playwright for DOM interaction and Ollama for reasoning.
Includes error recovery, state persistence, and notification support.
"""

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

# Browser automation
from browser_use import Agent
from langchain_ollama import ChatOllama

# Local database for tracking applications
from application_db import get_db, ApplicationStatus

# Cover letter generation
from cover_letter import get_cover_letter_instructions

# State management and error recovery
from agent_state import (
    get_state_manager,
    AgentState,
    AgentPhase,
    RetryConfig,
    calculate_retry_delay,
    is_retriable_error,
    is_browser_crash,
    is_captcha_error,
)

# Notifications
from notifications import (
    notify_captcha,
    notify_browser_crash,
    notify_network_error,
    notify_success,
    notify_error,
)

# Type definitions
LogCallback = Callable[[str, str], None]

# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    network_retries=5,
    browser_restart_retries=2,
)


@dataclass
class AgentResult:
    """Result of an agent run."""
    success: bool
    message: str
    application_id: Optional[int] = None
    fields_filled: int = 0
    research_queries: int = 0
    session_id: Optional[str] = None
    can_resume: bool = False


def load_profile(profile_path: str) -> str:
    """Load the user profile from markdown file."""
    path = Path(profile_path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Profile not found at: {profile_path}")


def build_system_prompt(profile_content: str, company: str = "the company", role: str = "the position") -> str:
    """Build the system prompt for the agent with the user's profile data."""
    
    # Get cover letter generation instructions
    cover_letter_instructions = get_cover_letter_instructions(profile_content, company, role)
    
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

{cover_letter_instructions}

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
    skip_duplicate_check: bool = False,
    retry_config: Optional[RetryConfig] = None,
    resume_session_id: Optional[str] = None
) -> dict:
    """
    Run the job application agent with error recovery and retry support.
    
    Args:
        job_url: The URL of the job posting/application page.
        resume_path: Path to the resume PDF file.
        profile_path: Path to the my_profile.md file.
        model_name: The Ollama model to use (default: qwen2.5:7b).
        log_callback: Optional callback function for logging (message, level).
        company: Company name (optional, for tracking).
        role: Role title (optional, for tracking).
        skip_duplicate_check: If True, skip duplicate URL check.
        retry_config: Configuration for retry behavior.
        resume_session_id: Session ID to resume from (for crash recovery).
    
    Returns:
        dict with 'success' (bool), 'message' (str), 'application_id' (int),
        'session_id' (str), and 'can_resume' (bool).
    """
    config = retry_config or DEFAULT_RETRY_CONFIG
    state_manager = get_state_manager()
    
    def log(message: str, level: str = "info"):
        """Log a message using the callback if provided."""
        if log_callback:
            log_callback(message, level)
        print(f"[{level.upper()}] {message}")
    
    db = get_db()
    application_id = None
    state: Optional[AgentState] = None
    
    try:
        # Check for session to resume
        if resume_session_id:
            state = state_manager.load_state(resume_session_id)
            if state:
                log(f"🔄 Resuming session: {state.session_id}", "info")
                application_id = state.application_id
                job_url = state.job_url
                company = state.company
                role = state.role
            else:
                log(f"⚠️ Could not load session {resume_session_id}", "warning")
        
        # Check for duplicate application
        if not skip_duplicate_check and not resume_session_id:
            existing = db.get_application_by_url(job_url)
            if existing:
                log(f"⚠️ Already applied to this job on {existing.created_at.strftime('%Y-%m-%d')}", "warning")
                return {
                    "success": False,
                    "message": f"Duplicate application: Already applied to {existing.company} - {existing.role}",
                    "application_id": existing.id,
                    "is_duplicate": True,
                    "can_resume": False
                }
        
        # Record application in database (if not resuming)
        if not application_id:
            application_id = db.add_application(
                company=company or "Unknown Company",
                role=role or "Unknown Role",
                job_url=job_url,
                status=ApplicationStatus.IN_PROGRESS,
                resume_used=resume_path
            )
        
        if application_id:
            log(f"📝 Application #{application_id} recorded in database", "info")
        
        # Create or update state
        if not state:
            state = state_manager.create_session(
                job_url=job_url,
                company=company or "Unknown",
                role=role or "Unknown",
                application_id=application_id,
                model_name=model_name,
                resume_path=resume_path,
                profile_path=profile_path
            )
            log(f"📋 Created session: {state.session_id}", "info")
        
        # Load profile
        log("Loading candidate profile...", "info")
        profile_content = load_profile(profile_path)
        log(f"Profile loaded ({len(profile_content)} characters)", "success")
        
        # Build system prompt with company/role for personalized cover letters
        system_prompt = build_system_prompt(
            profile_content, 
            company=company or "the company", 
            role=role or "the position"
        )
        
        # Initialize the LLM with retry
        log(f"Initializing Ollama with model: {model_name}", "info")
        llm = None
        llm_retry_count = 0
        
        while llm_retry_count < config.network_retries:
            try:
                llm = ChatOllama(
                    model=model_name,
                    temperature=0.1,
                    num_ctx=8192,
                )
                # Test connection
                log("LLM initialized successfully", "success")
                break
            except Exception as e:
                llm_retry_count += 1
                if llm_retry_count >= config.network_retries:
                    raise ConnectionError(f"Failed to connect to Ollama after {config.network_retries} attempts")
                
                delay = calculate_retry_delay(llm_retry_count - 1, config)
                log(f"⚠️ Ollama connection failed, retry {llm_retry_count}/{config.network_retries} in {delay:.1f}s...", "warning")
                notify_network_error(llm_retry_count, config.network_retries)
                state.record_error(str(e), "ollama_connection")
                state_manager.save_state(state)
                await asyncio.sleep(delay)
        
        # Update state
        state.update_phase(AgentPhase.NAVIGATING, "LLM ready, preparing browser")
        state_manager.save_state(state)
        
        # Initialize the browser agent with retry loop
        log("Launching browser...", "action")
        
        company_name = company or "the company"
        role_name = role or "the position"
        
        # Create the task for the agent
        task = f"""
NAVIGATE TO: {job_url}

YOUR GOAL: 
Complete the job application form up to the review stage. DO NOT SUBMIT.

APPLYING FOR: {role_name} at {company_name}

CANDIDATE PROFILE (Source of Truth):
{profile_content}

EXECUTION STEPS:
1. **Login/Start:** If a login is required, look for "Apply without account" or "Apply with Resume". If blocked by a complex login, PAUSE and ask user.
2. **Resume Upload:** Upload the file at: "{resume_path}".
3. **Form Filling:** - Map profile data to fields (e.g., "Experience" -> "Reynolds and Reynolds").
   - If a field asks for "Desired Salary" and it's not in the profile, OPEN A NEW TAB, search "average salary for [Role] at [Company]", extract the number, close tab, and fill it.
   - If a field asks "Why do you want to work here?", OPEN A NEW TAB, read the company's "About Us", and synthesize a short answer connecting their values to my "Thematic Interests" (in profile).

4. **COVER LETTER HANDLING:**
   If the application asks for a cover letter:
   
   a) **For TEXT FIELDS**: Write a short cover letter (3 paragraphs, under 300 words):
      
      BANNED phrases - never use: "I am writing to express", "I am excited", "leverage", "utilize", "passionate about"
      
      REQUIRED: Use contractions ("I'm" not "I am", "I've" not "I have")
      
      FORMAT:
      Dear {company_name} Team,
      
      [Why this {role_name} role caught your attention - 2 sentences]
      
      [2 specific things from your experience that fit - 3 sentences]  
      
      [You'd like to chat more - 1 sentence]
      
      Best,
      [Name from profile]
   
   b) **For FILE UPLOAD only**: Alert user that a cover letter file is needed.

5. **Review:** Click "Next" until you reach the "Review" or "Final Submit" page.
6. **HALT:** Stop immediately upon reaching the final page. notify the user: "Application ready for review."

CRITICAL RULES:
- **NO SUBMITTING:** Never click "Submit Application".
- **CAPTCHAS:** If you see a CAPTCHA, stop and alert the user.
- **PRIVACY:** Do not hallucinate personal data. Use *only* what is in the profile.
- **COVER LETTERS:** Use contractions, never say "I am excited" or "leverage". Keep under 300 words.
"""
        
        # Run agent with retry loop for browser crashes
        browser_retry_count = 0
        result = None
        
        while browser_retry_count <= config.browser_restart_retries:
            try:
                # Initialize and run the browser-use agent
                agent = Agent(
                    task=task,
                    llm=llm,  # type: ignore[arg-type]
                    extend_system_message=system_prompt,
                    browser_context_kwargs={
                        "headless": False,
                    }
                )
                
                log(f"Navigating to: {job_url}", "action")
                state.update_phase(AgentPhase.FILLING_FORM, f"Navigating to {job_url}")
                state_manager.save_state(state)
                
                # Run the agent
                result = await agent.run()
                
                # Check result for CAPTCHA
                result_str = str(result).lower() if result else ""
                if "captcha" in result_str:
                    log("🔐 CAPTCHA detected - please solve manually", "warning")
                    state.update_phase(AgentPhase.PAUSED_CAPTCHA, "CAPTCHA detected")
                    state_manager.save_state(state)
                    notify_captcha(company_name, job_url)
                    return {
                        "success": False,
                        "message": "CAPTCHA detected. Please solve it manually and retry.",
                        "application_id": application_id,
                        "session_id": state.session_id,
                        "can_resume": True
                    }
                
                # Success!
                break
                
            except Exception as e:
                error_msg = str(e)
                state.record_error(error_msg, type(e).__name__)
                
                # Check for CAPTCHA in error
                if is_captcha_error(e):
                    log("🔐 CAPTCHA detected - please solve manually", "warning")
                    state.update_phase(AgentPhase.PAUSED_CAPTCHA, "CAPTCHA detected")
                    state_manager.save_state(state)
                    notify_captcha(company_name, job_url)
                    return {
                        "success": False,
                        "message": "CAPTCHA detected. Please solve it manually and retry.",
                        "application_id": application_id,
                        "session_id": state.session_id,
                        "can_resume": True
                    }
                
                # Check if browser crashed and can retry
                if is_browser_crash(e) and browser_retry_count < config.browser_restart_retries:
                    browser_retry_count += 1
                    state.increment_retry()
                    delay = calculate_retry_delay(browser_retry_count - 1, config)
                    
                    log(f"⚠️ Browser crashed, restarting ({browser_retry_count}/{config.browser_restart_retries}) in {delay:.1f}s...", "warning")
                    notify_browser_crash(error_msg[:100])
                    state_manager.save_state(state)
                    
                    await asyncio.sleep(delay)
                    continue
                
                # Check if network error and can retry
                elif is_retriable_error(e) and state.retry_count < config.max_retries:
                    state.increment_retry()
                    delay = calculate_retry_delay(state.retry_count - 1, config)
                    
                    log(f"⚠️ Network error, retry {state.retry_count}/{config.max_retries} in {delay:.1f}s...", "warning")
                    notify_network_error(state.retry_count, config.max_retries)
                    state_manager.save_state(state)
                    
                    await asyncio.sleep(delay)
                    continue
                
                # Non-retriable error or max retries exceeded
                raise
        
        log("Agent completed its run", "success")
        
        # Update database and state to completed
        if application_id:
            db.update_status(application_id, ApplicationStatus.COMPLETED)
            log(f"✅ Application #{application_id} marked as completed", "success")
        
        state_manager.mark_completed(state)
        notify_success(company_name, role_name)
        
        return {
            "success": True,
            "message": "Agent completed. Please review the application before submitting.",
            "application_id": application_id,
            "session_id": state.session_id,
            "result": result,
            "can_resume": False
        }
        
    except FileNotFoundError as e:
        log(f"Profile file error: {e}", "error")
        if application_id:
            db.update_status(application_id, ApplicationStatus.FAILED, notes=str(e))
        if state:
            state_manager.mark_failed(state, str(e))
        notify_error(company or "Unknown", str(e))
        return {
            "success": False, 
            "message": str(e), 
            "application_id": application_id,
            "session_id": state.session_id if state else None,
            "can_resume": False
        }
    
    except ConnectionError as e:
        log("Could not connect to Ollama. Is it running?", "error")
        if application_id:
            db.update_status(application_id, ApplicationStatus.FAILED, notes="Ollama connection failed")
        if state:
            state_manager.mark_failed(state, "Ollama connection failed")
        notify_error(company or "Unknown", "Ollama connection failed")
        return {
            "success": False, 
            "message": "Ollama connection failed. Please ensure Ollama is running (ollama serve).",
            "application_id": application_id,
            "session_id": state.session_id if state else None,
            "can_resume": True  # Can retry after starting Ollama
        }
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log(f"Unexpected error: {error_msg}", "error")
        
        if application_id:
            db.update_status(application_id, ApplicationStatus.FAILED, notes=error_msg)
        
        # Determine if session can be resumed
        can_resume = False
        if state:
            if is_retriable_error(e) or is_browser_crash(e):
                can_resume = True
                state.update_phase(AgentPhase.PAUSED_USER, f"Error: {error_msg}")
            else:
                state_manager.mark_failed(state, error_msg)
            state_manager.save_state(state)
        
        notify_error(company or "Unknown", str(e)[:100])
        
        return {
            "success": False, 
            "message": f"Agent error: {str(e)}", 
            "application_id": application_id,
            "session_id": state.session_id if state else None,
            "can_resume": can_resume
        }


async def resume_agent(session_id: str, log_callback: Optional[LogCallback] = None) -> dict:
    """
    Resume an interrupted agent session.
    
    Args:
        session_id: The session ID to resume
        log_callback: Optional logging callback
    
    Returns:
        Result dict from run_agent
    """
    state_manager = get_state_manager()
    state = state_manager.load_state(session_id)
    
    if not state:
        return {
            "success": False,
            "message": f"Session {session_id} not found",
            "can_resume": False
        }
    
    return await run_agent(
        job_url=state.job_url,
        resume_path=state.resume_path,
        profile_path=state.profile_path,
        model_name=state.model_name,
        log_callback=log_callback,
        company=state.company,
        role=state.role,
        skip_duplicate_check=True,
        resume_session_id=session_id
    )


def get_recoverable_sessions() -> list:
    """
    Get list of sessions that can be recovered.
    
    Returns:
        List of AgentState objects for recoverable sessions
    """
    state_manager = get_state_manager()
    return state_manager.get_recoverable_sessions()


def check_active_session() -> Optional[AgentState]:
    """
    Check if there's an active session that was interrupted.
    
    Returns:
        AgentState if found, None otherwise
    """
    state_manager = get_state_manager()
    return state_manager.get_active_session()


# For testing the agent directly
if __name__ == "__main__":
    import sys
    
    def print_usage():
        print("\nUsage:")
        print("  python apply_agent.py <job_url> <resume_path>  - Run agent for a job")
        print("  python apply_agent.py resume <session_id>      - Resume a session")
        print("  python apply_agent.py sessions                 - List recoverable sessions")
        print("  python apply_agent.py check                    - Check for active session")
        print("\nExample:")
        print("  python apply_agent.py https://boards.greenhouse.io/company/jobs/123 /path/to/resume.pdf")
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "sessions":
        # List recoverable sessions
        sessions = get_recoverable_sessions()
        if sessions:
            print(f"\n📋 Recoverable Sessions ({len(sessions)}):\n")
            for s in sessions:
                print(f"  Session: {s.session_id}")
                print(f"    Job: {s.role} at {s.company}")
                print(f"    URL: {s.job_url[:60]}...")
                print(f"    Phase: {s.phase.value}")
                print(f"    Retries: {s.retry_count}")
                print(f"    Updated: {s.updated_at}")
                print()
        else:
            print("No recoverable sessions found.")
    
    elif command == "check":
        # Check for active session
        state = check_active_session()
        if state:
            print(f"\n🔄 Active Session Found:")
            print(f"  Session: {state.session_id}")
            print(f"  Job: {state.role} at {state.company}")
            print(f"  URL: {state.job_url}")
            print(f"  Phase: {state.phase.value}")
            print(f"\nTo resume: python apply_agent.py resume {state.session_id}")
        else:
            print("No active session found.")
    
    elif command == "resume":
        if len(sys.argv) < 3:
            print("Error: Session ID required")
            print("Usage: python apply_agent.py resume <session_id>")
            sys.exit(1)
        
        session_id = sys.argv[2]
        print(f"Resuming session: {session_id}")
        
        result = asyncio.run(resume_agent(session_id))
        print(f"\nResult: {result}")
    
    elif command.startswith("http"):
        # Run agent for a job URL
        if len(sys.argv) < 3:
            print("Error: Resume path required")
            print_usage()
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
        
        if result.get("can_resume") and result.get("session_id"):
            print(f"\n💡 To resume: python apply_agent.py resume {result['session_id']}")
    
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)