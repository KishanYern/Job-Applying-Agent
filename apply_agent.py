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
from typing import Callable, Optional, Any
from dataclasses import dataclass

# Browser automation
from browser_use import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.messages.tool import ToolCall
import json
import copy

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


def _extract_candidate_name(profile_content: str) -> str:
    """Extract the candidate's name from the profile using a robust regex."""
    import re
    match = re.search(r'^Name:\s*(.+)$', profile_content, re.MULTILINE)
    return match.group(1).strip() if match else "the candidate"


def build_system_prompt(
    profile_content: str,
    company_name: str,
    company_info: Optional[Any] = None,
    job_reqs: Optional[Any] = None,
    why_here_answer: Optional[str] = None,
) -> str:
    """
    Build the Phase 2 system prompt with all pre-researched data injected.
    """
    # Extract pre-researched fields
    company_data_lines = []
    if company_info:
        for label, value in [
            ("Company Mission", company_info.values_summary),
            ("About", company_info.about_summary),
            ("Recent News", company_info.recent_news),
        ]:
            if value:
                company_data_lines.append(f"- **{label}:** {value}")

    company_section = (
        "\n".join(company_data_lines) if company_data_lines
        else "No pre-researched company data available."
    )

    job_req_lines = []
    if job_reqs:
        for label, value in [
            ("Role Description", job_reqs.tech_stack),
            ("Skills Required", job_reqs.skills_required),
            ("Salary Range", job_reqs.salary_range),
        ]:
            if value:
                job_req_lines.append(f"- **{label}:** {value}")

    job_section = (
        "\n".join(job_req_lines) if job_req_lines
        else "No pre-extracted job requirements available."
    )

    why_here = why_here_answer or ""
    cover_letter = (job_reqs.cover_letter_text if job_reqs else None) or ""
    cover_letter_pdf_path = (job_reqs.cover_letter_pdf_path if job_reqs else None) or ""

    # Only expose the filename, not the full local path, to the remote LLM
    from pathlib import Path as _Path
    cover_letter_pdf_display = _Path(cover_letter_pdf_path).name if cover_letter_pdf_path else ""

    # Build optional sections
    optional_sections = []

    if why_here:
        optional_sections.append(f"## \"WHY DO YOU WANT TO WORK HERE?\" ANSWER (pre-synthesized)\n{why_here}")

    if cover_letter:
        optional_sections.append(f"## PRE-GENERATED COVER LETTER\n{cover_letter}")

    if cover_letter_pdf_display:
        optional_sections.append(f"## COVER LETTER PDF FILE\n{cover_letter_pdf_display}")

    optional_block = "\n\n".join(optional_sections)

    candidate_name = _extract_candidate_name(profile_content)

    return f"""You are an autonomous job application agent. Fill out job application forms accurately using the candidate profile and pre-researched data below.

## PRE-RESEARCHED COMPANY DATA
{company_section}

## JOB REQUIREMENTS
{job_section}

{optional_block}

## FORM FILLING RULES
1. Map form labels to the CANDIDATE PROFILE provided in the task. Use exact values — never paraphrase names, emails, or phone numbers.
2. Handle dropdowns by selecting the closest matching option.
3. For resume uploads, use the file selector with the path given in the task.
4. For salary fields, use the Salary Range above. If unavailable, call search_missing_info.
5. For "Why here" fields, use the pre-synthesized answer above verbatim or lightly adapted.
6. For cover letter text fields, paste the Pre-Generated Cover Letter. If unavailable, write one from the profile and company data.
7. For cover letter file uploads, upload the PDF listed above. If unavailable, alert the user.
8. DO NOT open new browser tabs. Use search_missing_info for missing data.
9. For unforeseen behavioral or subjective questions, you MUST construct your response using only facts from the CANDIDATE PROFILE and the pre-researched context provided above (company mission, role description, recent news, etc.). Never invent information.

## SAFETY RULES
- NEVER submit the application without user confirmation.
- STOP and alert if you detect a CAPTCHA, login requirement, or are uncertain about a field.
- Wait for pages to load fully before interacting.

You are applying as: {candidate_name}
"""


class SafeVLLMChatOpenAI(ChatOpenAI):
    """
    Bypasses LangChain's OpenAI HTTP client entirely to avoid RunPod/vLLM 2.14.0
    crashing on extra OpenAI-native fields (logprobs, n, frequency_penalty, etc.)
    that LangChain silently injects. Instead we fire a raw requests/httpx call
    with only the minimal payload that the scratch test proved works:
    {model, messages, max_tokens, temperature}. Tools are injected into the
    system prompt as XML text; <tool_call> blocks in responses are parsed back
    into standard LangChain ToolCall objects.
    """

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_raw_messages(self, messages, tools=None):
        def to_dict(msg):
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
            if isinstance(msg, SystemMessage):
                return {"role": "system", "content": msg.content}
            elif isinstance(msg, HumanMessage):
                return {"role": "user", "content": msg.content}
            elif isinstance(msg, AIMessage):
                content = msg.content or ""
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content += f'\n<tool_call>\n{{"name": "{tc["name"]}", "arguments": {json.dumps(tc["args"])}}}\n</tool_call>\n'
                return {"role": "assistant", "content": content.strip()}
            elif isinstance(msg, ToolMessage):
                return {"role": "user", "content": f"<tool_response>\nName: {msg.name}\nContent: {msg.content}\n</tool_response>"}
            else:
                return {"role": "user", "content": str(msg.content)}

        raw = [to_dict(m) for m in messages]

        if tools:
            instr = (
                "\n\nYou are a helpful assistant with tool calling capabilities. "
                "You must use the provided tools if needed.\n"
                "<tools>\n" + json.dumps(tools, indent=2) + "\n</tools>\n"
                'If you want to use a tool, output EXACTLY:\n'
                "<tool_call>\n"
                '{"name": "<tool_name>", "arguments": <json_args>}\n'
                "</tool_call>\n"
            )
            if raw and raw[0]["role"] == "system":
                raw[0]["content"] += instr
            else:
                raw.insert(0, {"role": "system", "content": instr})
        return raw

    @staticmethod
    def _parse_tool_calls(text):
        import re
        parsed = []
        blocks = list(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL))
        for i, m in enumerate(reversed(blocks)):
            try:
                blob = json.loads(m.group(1))
                args = blob.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                parsed.insert(0, ToolCall(
                    name=blob["name"], args=args,
                    id=f"call_{int(time.time()*1000)}_{i}",
                ))
                text = text[:m.start()] + text[m.end():]
            except Exception as exc:
                print(f"[WARN] SafeVLLM: failed to parse tool call: {exc}")
        return text.strip(), parsed

    def _build_payload(self, raw_messages, stop):
        payload = {
            "model": self.model_name,
            "messages": raw_messages,
            "max_tokens": self.max_tokens or 8192,
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop
        return payload

    def _headers(self):
        key = (self.openai_api_key.get_secret_value()
               if hasattr(self.openai_api_key, "get_secret_value")
               else str(self.openai_api_key))
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    @property
    def _chat_url(self):
        base = str(self.openai_api_base or "").rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    @staticmethod
    def _to_chat_result(data):
        from langchain_core.outputs import ChatGeneration
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason", "stop")
        ai_msg = AIMessage(content=content)
        gen = ChatGeneration(message=ai_msg, generation_info={"finish_reason": finish})
        return ChatResult(
            generations=[gen],
            llm_output={"token_usage": data.get("usage", {}), "model_name": data.get("model", "")},
        )

    # ------------------------------------------------------------------ #
    #  Sync                                                                #
    # ------------------------------------------------------------------ #

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import requests as _req
        tools = kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("parallel_tool_calls", None)

        payload = self._build_payload(self._build_raw_messages(messages, tools), stop)
        timeout = getattr(self, "request_timeout", None) or getattr(self, "timeout", 800)

        resp = _req.post(self._chat_url, headers=self._headers(), json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise Exception(f"Error code: {resp.status_code} - {resp.json()}")

        result = self._to_chat_result(resp.json())
        if tools and result.generations:
            ai = result.generations[0].message
            ai.content, ai.tool_calls = self._parse_tool_calls(ai.content or "")
        return result

    # ------------------------------------------------------------------ #
    #  Async                                                               #
    # ------------------------------------------------------------------ #

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        import httpx
        tools = kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("parallel_tool_calls", None)

        payload = self._build_payload(self._build_raw_messages(messages, tools), stop)
        timeout = float(getattr(self, "request_timeout", None) or getattr(self, "timeout", 800))

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self._chat_url, headers=self._headers(), json=payload)
        if resp.status_code != 200:
            raise Exception(f"Error code: {resp.status_code} - {resp.json()}")

        result = self._to_chat_result(resp.json())
        if tools and result.generations:
            ai = result.generations[0].message
            ai.content, ai.tool_calls = self._parse_tool_calls(ai.content or "")
        return result

def _init_llm(model_name: Optional[str] = None) -> SafeVLLMChatOpenAI:
    """
    Initialize the SafeVLLMChatOpenAI client pointed at the remote cloud GPU vLLM endpoint.
    Reads RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY from environment variables.
    """
    endpoint_url = os.environ.get("RUNPOD_ENDPOINT_URL", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    model_name = model_name or os.environ.get("RUNPOD_MODEL_NAME", "casperhansen/llama-3.3-70b-instruct-awq")

    if not endpoint_url or not api_key:
        raise ConnectionError(
            "RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY must be set to connect to the cloud GPU. "
            "Configure them in the UI or set them as environment variables."
        )

    return SafeVLLMChatOpenAI(
        model=model_name,
        base_url=endpoint_url,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
        max_retries=10,
        timeout=800,
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

        # Retrieve company info and job requirements from ApplicationDB
        company_name_query = company or "the company"
        company_info = db.get_company_info(company_name_query) if company_name_query != "the company" else None
        job_reqs = db.get_job_requirements(application_id) if application_id else None
        why_here_db = db.find_answer("why do you want to work here", company_name_query)
        why_here_answer = why_here_db.answer if why_here_db else None

        # Build system prompt with all pre-researched data
        system_prompt = build_system_prompt(
            profile_content,
            company_name=company_name_query,
            company_info=company_info,
            job_reqs=job_reqs,
            why_here_answer=why_here_answer,
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

        # Cover letter PDF: agent needs the full local path to upload the file,
        # but system prompt only shows the filename for privacy. Provide full
        # path only in the task message which stays within the agent session.
        cover_letter_pdf_full = ""
        if job_id:
            _jreqs = db.get_job_requirements(job_id)
            if _jreqs and _jreqs.cover_letter_pdf_path:
                cover_letter_pdf_full = _jreqs.cover_letter_pdf_path

        task = f"""
NAVIGATE TO: {job_url}

GOAL: Complete the job application form up to the review stage. DO NOT SUBMIT.

APPLYING FOR: {role_name} at {company_name}

## CANDIDATE PROFILE
{profile_content}

EXECUTION STEPS:
1. If a login is required, look for "Apply without account". If blocked, PAUSE and ask the user.
2. Upload the resume at: "{resume_path}".
3. Fill all form fields using the CANDIDATE PROFILE above and the pre-researched data in the system prompt.
4. For cover letter file uploads, use: "{cover_letter_pdf_full}" (if empty, alert the user).
5. Click "Next" / "Continue" until you reach the review page.
6. HALT immediately. Notify the user: "Application ready for review."
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
                        "title", "type", "name", "role", "id", "aria-label",
                        "placeholder", "value",
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

        # Basic validation: check that the result looks like real progress
        result_str = str(result).lower() if result else ""
        likely_success = any(
            signal in result_str
            for signal in ["review", "ready", "filled", "complete", "submitted", "next"]
        )

        if application_id:
            final_status = ApplicationStatus.COMPLETED if likely_success else ApplicationStatus.IN_PROGRESS
            db.update_status(application_id, final_status)
            if likely_success:
                log(f"Application #{application_id} marked as completed", "success")
            else:
                log(
                    f"Application #{application_id} finished but could not confirm progress — "
                    f"marked as in_progress for manual review",
                    "warning",
                )

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
