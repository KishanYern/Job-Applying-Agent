"""
AI Job Application Agent V2 - Core Agent Logic
Phase 2: Browser automation + form-filling powered by Qwen2.5-Coder-32B
running on a remote cloud GPU (RunPod / Vast.ai) via OpenAI-compatible vLLM endpoint.

All company research, salary data, and cover letters are pre-loaded from the SQLite
cache populated by Phase 1 (job_scraper.py). The browser agent never opens new tabs
to Google — it works entirely from cached data plus an on-the-spot Tavily fallback tool.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

# Browser automation
from browser_use import Agent
from langchain_openai import ChatOpenAI

# Local database for tracking applications
from application_db import get_db, ApplicationStatus

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
    session_id: Optional[str] = None
    can_resume: bool = False


def load_profile(profile_path: str) -> str:
    """Load the user profile from markdown file."""
    path = Path(profile_path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Profile not found at: {profile_path}")


def _build_search_tools(company: str):
    """
    Build a browser-use Tools object with a Tavily search action registered.
    Passed to Agent(tools=...) so the 32B model can call it when cached data is missing.
    Returns None if browser-use Tools API is unavailable (graceful degradation).
    """
    try:
        from browser_use import Tools, ActionResult
        from application_db import get_db as _get_db

        tools = Tools()

        @tools.action(
            description=(
                "Search the web for specific information that is missing from the "
                "pre-researched cache (e.g. salary data, company details). "
                "Use this ONLY when the required data is marked 'Not available' in the "
                "system prompt. Do NOT open new browser tabs — this queries an API directly."
            )
        )
        async def search_missing_info(query: str) -> ActionResult:
            tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
            if not tavily_api_key:
                return ActionResult(extracted_content="Search unavailable: TAVILY_API_KEY not configured.")
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_api_key)
                result = client.search(query, max_results=3)
                answer = result.get("results", [{}])[0].get("content", "No results found.")
                db = _get_db()
                db.save_answer(query, answer, company)
                return ActionResult(extracted_content=answer)
            except Exception as e:
                return ActionResult(extracted_content=f"Search failed: {e}")

        return tools

    except (ImportError, Exception) as e:
        print(f"[WARN] Could not register search tool: {e}")
        return None


def build_system_prompt(
    profile_content: str,
    company: str = "the company",
    role: str = "the position",
    job_id: Optional[int] = None,
) -> str:
    """
    Build the Phase 2 system prompt with all pre-researched data injected.
    Queries the SQLite DB for cached company info, job requirements, and answers.
    """
    db = get_db()
    company_info = db.get_company_info(company) if company != "the company" else None
    job_reqs = db.get_job_requirements(job_id) if job_id else None
    why_here_answer = db.find_answer("why do you want to work here", company)

    # Extract pre-researched fields with safe fallbacks
    values_summary = (company_info.values_summary if company_info else None) or "Not available"
    about_summary = (company_info.about_summary if company_info else None) or "Not available"
    competitors = (company_info.competitors if company_info else None) or "Not available"
    recent_news = (company_info.recent_news if company_info else None) or "Not available"
    tech_stack = (job_reqs.tech_stack if job_reqs else None) or "Not available"
    skills_required = (job_reqs.skills_required if job_reqs else None) or "Not available"
    salary_range = (job_reqs.salary_range if job_reqs else None) or "Not available"
    cover_letter = (job_reqs.cover_letter_text if job_reqs else None) or ""
    cover_letter_pdf_path = (job_reqs.cover_letter_pdf_path if job_reqs else None) or ""
    why_here = (why_here_answer.answer if why_here_answer else None) or "Not available"

    cover_letter_section = (
        f"## PRE-GENERATED COVER LETTER\n{cover_letter}"
        if cover_letter
        else "## PRE-GENERATED COVER LETTER\nNot available — write one using the profile and company data above."
    )

    cover_letter_pdf_section = (
        f"## COVER LETTER PDF FILE\n{cover_letter_pdf_path}"
        if cover_letter_pdf_path
        else "## COVER LETTER PDF FILE\nNot available"
    )

    candidate_name = (
        profile_content.split("Name:")[1].split("\n")[0].strip()
        if "Name:" in profile_content
        else "the candidate"
    )

    return f"""You are an autonomous job application agent. Your task is to fill out job application forms accurately and professionally using the candidate's profile and the pre-researched data provided below.

## CANDIDATE PROFILE (Source of Truth for all personal data)
{profile_content}

## PRE-RESEARCHED COMPANY DATA (from Phase 1 discovery pipeline)
- **Values / Mission:** {values_summary}
- **About / Products:** {about_summary}
- **Competitors:** {competitors}
- **Recent News:** {recent_news}

## JOB REQUIREMENTS (extracted from job description)
- **Tech Stack:** {tech_stack}
- **Skills Required:** {skills_required}
- **Salary Range:** {salary_range}

## "WHY DO YOU WANT TO WORK HERE?" ANSWER (pre-synthesized)
{why_here}

{cover_letter_section}

{cover_letter_pdf_section}

## FORM FILLING RULES
1. **Match fields carefully**: Map form labels (e.g., "First Name", "Email", "Phone") to the candidate profile.
2. **Be precise**: Use exact values from the profile. Do not paraphrase names, emails, or phone numbers.
3. **Handle dropdowns**: Select the closest matching option.
4. **File uploads**: If asked to upload a resume, use the file selector.
5. **Salary fields**: Use the pre-researched Salary Range above. If it is "Not available", use the search_missing_info tool.
6. **"Why here" fields**: Use the pre-synthesized answer above verbatim or lightly adapted.
7. **Cover letter text fields**: Paste the Pre-Generated Cover Letter above. If it is not available, write one from the profile and company data.
8. **Cover letter file uploads**: If a cover letter file upload field is detected, upload the PDF file listed in the COVER LETTER PDF FILE section above. If that path is "Not available", alert the user that a cover letter file is needed.
9. **DO NOT open new browser tabs** for any reason. Use the search_missing_info tool for any missing data.
10. **Unforeseen short-answer or essay questions**: If any form field asks a behavioral, technical, or subjective question not covered above (e.g., "Tell us about a challenging project", "Describe a complex problem you solved"), compose a concise, professional answer (2-4 sentences) using ONLY facts, experiences, and skills explicitly stated in the CANDIDATE PROFILE section above. Never invent experiences, metrics, or projects not present in the profile. If the profile contains no relevant information for the question, enter: "Please see my attached resume for details."

## CRITICAL SAFETY RULES
1. **NEVER submit** the application without explicit user confirmation.
2. **STOP and alert** if you detect a CAPTCHA — do not attempt to solve it.
3. **STOP and alert** if a login is required — the user must handle authentication.
4. **STOP and alert** if you are uncertain about a field — do not guess personal data.
5. **Be patient**: Wait for pages to load fully before interacting.

## NAVIGATION TIPS
- Look for "Apply", "Submit Application", or similar buttons to start.
- Handle multi-page applications by clicking "Next" or "Continue".
- Watch for confirmation messages to know when sections are complete.

You are applying as: {candidate_name}
"""


def _init_llm() -> ChatOpenAI:
    """
    Initialize the ChatOpenAI client pointed at the remote cloud GPU vLLM endpoint.
    Reads RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY from environment variables.
    """
    endpoint_url = os.environ.get("RUNPOD_ENDPOINT_URL", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")

    if not endpoint_url or not api_key:
        raise ConnectionError(
            "RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY must be set to connect to the cloud GPU. "
            "Configure them in the UI or set them as environment variables."
        )

    return ChatOpenAI(
        model="qwen2.5-coder-32b-instruct",
        base_url=endpoint_url,
        api_key=api_key,
        temperature=0.1,
        max_tokens=8192,
    )


async def run_agent(
    job_url: str,
    resume_path: str,
    profile_path: str,
    log_callback: Optional[LogCallback] = None,
    company: Optional[str] = None,
    role: Optional[str] = None,
    job_id: Optional[int] = None,
    skip_duplicate_check: bool = False,
    retry_config: Optional[RetryConfig] = None,
    resume_session_id: Optional[str] = None,
) -> dict:
    """
    Run the Phase 2 job application agent.

    Args:
        job_url: The URL of the job posting/application page.
        resume_path: Path to the resume PDF file.
        profile_path: Path to the my_profile.md file.
        log_callback: Optional callback function for logging (message, level).
        company: Company name (optional, for tracking and DB lookup).
        role: Role title (optional, for tracking).
        job_id: Application DB ID used to retrieve cached job requirements.
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
        if log_callback:
            log_callback(message, level)
        print(f"[{level.upper()}] {message}")

    db = get_db()
    application_id = job_id
    state: Optional[AgentState] = None

    try:
        # Resume an existing session
        if resume_session_id:
            state = state_manager.load_state(resume_session_id)
            if state:
                log(f"Resuming session: {state.session_id}", "info")
                application_id = state.application_id
                job_url = state.job_url
                company = state.company
                role = state.role
            else:
                log(f"Could not load session {resume_session_id}", "warning")

        # Duplicate check (skip if we already have a job_id from Phase 1)
        if not skip_duplicate_check and not resume_session_id and not application_id:
            existing = db.get_application_by_url(job_url)
            if existing:
                log(
                    f"Already applied to this job on {existing.created_at.strftime('%Y-%m-%d')}",
                    "warning",
                )
                return {
                    "success": False,
                    "message": f"Duplicate application: Already applied to {existing.company} - {existing.role}",
                    "application_id": existing.id,
                    "is_duplicate": True,
                    "can_resume": False,
                }

        # Register application if not already done by Phase 1
        if not application_id:
            application_id = db.add_application(
                company=company or "Unknown Company",
                role=role or "Unknown Role",
                job_url=job_url,
                status=ApplicationStatus.IN_PROGRESS,
                resume_used=resume_path,
            )
        else:
            db.update_status(application_id, ApplicationStatus.IN_PROGRESS)

        if application_id:
            log(f"Application #{application_id} in progress", "info")

        # Create or restore session state
        if not state:
            state = state_manager.create_session(
                job_url=job_url,
                company=company or "Unknown",
                role=role or "Unknown",
                application_id=application_id,
                model_name="qwen2.5-coder-32b-instruct",
                resume_path=resume_path,
                profile_path=profile_path,
            )
            log(f"Session created: {state.session_id}", "info")

        # Load profile
        log("Loading candidate profile...", "info")
        profile_content = load_profile(profile_path)
        log(f"Profile loaded ({len(profile_content)} characters)", "success")

        # Build system prompt with all pre-researched data
        system_prompt = build_system_prompt(
            profile_content,
            company=company or "the company",
            role=role or "the position",
            job_id=application_id,
        )

        # Initialize LLM with retry
        log("Connecting to cloud GPU endpoint...", "info")
        llm = None
        llm_retry_count = 0

        while llm_retry_count < config.network_retries:
            try:
                llm = _init_llm()
                log("Cloud GPU endpoint connected", "success")
                break
            except ConnectionError as e:
                raise  # Missing env vars — no point retrying
            except Exception as e:
                llm_retry_count += 1
                if llm_retry_count >= config.network_retries:
                    raise ConnectionError(
                        f"Failed to connect to cloud GPU after {config.network_retries} attempts"
                    )
                delay = calculate_retry_delay(llm_retry_count - 1, config)
                log(
                    f"GPU endpoint unavailable, retry {llm_retry_count}/{config.network_retries} "
                    f"in {delay:.1f}s...",
                    "warning",
                )
                notify_network_error(llm_retry_count, config.network_retries)
                state.record_error(str(e), "gpu_connection")
                state_manager.save_state(state)
                await asyncio.sleep(delay)

        state.update_phase(AgentPhase.NAVIGATING, "LLM ready, preparing browser")
        state_manager.save_state(state)

        log("Launching browser...", "action")

        company_name = company or "the company"
        role_name = role or "the position"

        # Build on-the-spot Tavily fallback tool (browser-use Tools object)
        search_tools = _build_search_tools(company_name)

        task = f"""
NAVIGATE TO: {job_url}

YOUR GOAL:
Complete the job application form up to the review stage. DO NOT SUBMIT.

APPLYING FOR: {role_name} at {company_name}

EXECUTION STEPS:
1. **Login / Start:** If a login is required, look for "Apply without account". If blocked by a complex login, PAUSE and ask the user.
2. **Resume Upload:** Upload the file at: "{resume_path}".
3. **Form Filling:**
   - Map profile data to each field using the CANDIDATE PROFILE in the system prompt.
   - For salary fields: use the pre-researched Salary Range. If "Not available", call search_missing_info.
   - For "Why do you want to work here?": use the PRE-SYNTHESIZED ANSWER from the system prompt.
   - For cover letter text fields: paste the PRE-GENERATED COVER LETTER from the system prompt.
   - For cover letter file uploads: upload the PDF file specified in the COVER LETTER PDF FILE section of the system prompt. If the path is "Not available", alert the user.
   - For any unforeseen behavioral, technical, or subjective questions: compose a short answer (2-4 sentences) using ONLY facts from the CANDIDATE PROFILE. Never invent information. If nothing relevant is in the profile, write: "Please see my attached resume for details."
   - DO NOT open new browser tabs for any reason.
4. **Review:** Click "Next" until you reach the "Review" or "Final Submit" page.
5. **HALT:** Stop immediately upon reaching the final page. Notify the user: "Application ready for review."

CRITICAL RULES:
- NEVER click "Submit Application".
- If you see a CAPTCHA, stop and alert the user.
- Only use data from the CANDIDATE PROFILE — do not hallucinate personal information.
- Use the search_missing_info tool (not the browser) for any data gaps.
"""

        # Run agent with browser crash retry loop
        browser_retry_count = 0
        result = None

        while browser_retry_count <= config.browser_restart_retries:
            try:
                agent_kwargs: dict = dict(
                    task=task,
                    llm=llm,  # type: ignore[arg-type]
                    extend_system_message=system_prompt,
                    include_attributes=[
                        "title", "type", "name", "role", "aria-label",
                        "placeholder", "value", "alt", "for", "href",
                    ],
                    browser_context_kwargs={"headless": False},
                )
                if search_tools is not None:
                    agent_kwargs["tools"] = search_tools

                agent = Agent(**agent_kwargs)

                log(f"Navigating to: {job_url}", "action")
                state.update_phase(AgentPhase.FILLING_FORM, f"Navigating to {job_url}")
                state_manager.save_state(state)

                result = await agent.run()

                # CAPTCHA detection
                result_str = str(result).lower() if result else ""
                if "captcha" in result_str:
                    log("CAPTCHA detected — please solve manually", "warning")
                    state.update_phase(AgentPhase.PAUSED_CAPTCHA, "CAPTCHA detected")
                    state_manager.save_state(state)
                    notify_captcha(company_name, job_url)
                    return {
                        "success": False,
                        "message": "CAPTCHA detected. Please solve it manually and retry.",
                        "application_id": application_id,
                        "session_id": state.session_id,
                        "can_resume": True,
                    }

                break  # Success

            except Exception as e:
                error_msg = str(e)
                state.record_error(error_msg, type(e).__name__)

                if is_captcha_error(e):
                    log("CAPTCHA detected — please solve manually", "warning")
                    state.update_phase(AgentPhase.PAUSED_CAPTCHA, "CAPTCHA detected")
                    state_manager.save_state(state)
                    notify_captcha(company_name, job_url)
                    return {
                        "success": False,
                        "message": "CAPTCHA detected. Please solve manually and retry.",
                        "application_id": application_id,
                        "session_id": state.session_id,
                        "can_resume": True,
                    }

                if is_browser_crash(e) and browser_retry_count < config.browser_restart_retries:
                    browser_retry_count += 1
                    state.increment_retry()
                    delay = calculate_retry_delay(browser_retry_count - 1, config)
                    log(
                        f"Browser crashed, restarting ({browser_retry_count}/{config.browser_restart_retries})"
                        f" in {delay:.1f}s...",
                        "warning",
                    )
                    notify_browser_crash(error_msg[:100])
                    state_manager.save_state(state)
                    await asyncio.sleep(delay)
                    continue

                elif is_retriable_error(e) and state.retry_count < config.max_retries:
                    state.increment_retry()
                    delay = calculate_retry_delay(state.retry_count - 1, config)
                    log(
                        f"Network error, retry {state.retry_count}/{config.max_retries} in {delay:.1f}s...",
                        "warning",
                    )
                    notify_network_error(state.retry_count, config.max_retries)
                    state_manager.save_state(state)
                    await asyncio.sleep(delay)
                    continue

                raise

        log("Agent completed its run", "success")

        if application_id:
            db.update_status(application_id, ApplicationStatus.COMPLETED)
            log(f"Application #{application_id} marked as completed", "success")

        state_manager.mark_completed(state)
        notify_success(company_name, role_name)

        return {
            "success": True,
            "message": "Agent completed. Please review the application before submitting.",
            "application_id": application_id,
            "session_id": state.session_id,
            "result": result,
            "can_resume": False,
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
            "can_resume": False,
        }

    except ConnectionError as e:
        log(str(e), "error")
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
            "can_resume": True,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log(f"Unexpected error: {error_msg}", "error")

        if application_id:
            db.update_status(application_id, ApplicationStatus.FAILED, notes=error_msg)

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
            "can_resume": can_resume,
        }


async def resume_agent(
    session_id: str, log_callback: Optional[LogCallback] = None
) -> dict:
    """Resume an interrupted agent session."""
    state_manager = get_state_manager()
    state = state_manager.load_state(session_id)

    if not state:
        return {
            "success": False,
            "message": f"Session {session_id} not found",
            "can_resume": False,
        }

    return await run_agent(
        job_url=state.job_url,
        resume_path=state.resume_path,
        profile_path=state.profile_path,
        log_callback=log_callback,
        company=state.company,
        role=state.role,
        job_id=state.application_id,
        skip_duplicate_check=True,
        resume_session_id=session_id,
    )


def get_recoverable_sessions() -> list:
    """Get list of sessions that can be recovered."""
    return get_state_manager().get_recoverable_sessions()


def check_active_session() -> Optional[AgentState]:
    """Check if there's an active session that was interrupted."""
    return get_state_manager().get_active_session()


# CLI entry point
if __name__ == "__main__":
    import sys

    def print_usage():
        print("\nUsage:")
        print("  python apply_agent.py <job_url> <resume_path>  - Run agent for a job")
        print("  python apply_agent.py resume <session_id>      - Resume a session")
        print("  python apply_agent.py sessions                 - List recoverable sessions")
        print("  python apply_agent.py check                    - Check for active session")

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "sessions":
        sessions = get_recoverable_sessions()
        if sessions:
            print(f"\nRecoverable Sessions ({len(sessions)}):\n")
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
        state = check_active_session()
        if state:
            print(f"\nActive Session Found:")
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
            sys.exit(1)
        session_id = sys.argv[2]
        print(f"Resuming session: {session_id}")
        result = asyncio.run(resume_agent(session_id))
        print(f"\nResult: {result}")

    elif command.startswith("http"):
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

        result = asyncio.run(
            run_agent(
                job_url=job_url,
                resume_path=resume_path,
                profile_path=str(profile_path),
            )
        )

        print(f"\nResult: {result}")

        if result.get("can_resume") and result.get("session_id"):
            print(f"\nTo resume: python apply_agent.py resume {result['session_id']}")

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
