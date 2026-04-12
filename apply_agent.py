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
import re
import time
from pathlib import Path
from typing import Callable, Optional, Any

# Load .env so credentials are available both from CLI and when imported by app.py
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass

# Browser automation
from browser_use import Agent
from browser_use.browser.session import BrowserSession
from browser_use import BrowserProfile
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

from runpod_workers import managed_runpod_workers

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


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse truthy env vars (1/true/yes/on)."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _build_search_tools(company: str):
    """
    Build a browser-use Tools object with custom actions registered.
    Passed to Agent(tools=...) so the model can call specialised helpers.
    Returns None if browser-use Tools API is unavailable (graceful degradation).
    """
    try:
        from browser_use import Tools, ActionResult
        from browser_use.browser.session import BrowserSession
        from application_db import get_db as _get_db

        tools = Tools()

        # ------------------------------------------------------------------ #
        # 1. Tavily web-search fallback                                       #
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # 2. Typeahead / searchable-combobox filler                          #
        # Use this for Greenhouse School, Degree, Discipline, Country, etc.  #
        # It types a short prefix, waits for the ARIA listbox, and clicks    #
        # the first option whose text contains `value` (case-insensitive).   #
        # Falls back to clicking the first option if no exact match found.   #
        # ------------------------------------------------------------------ #
        @tools.action(
            description=(
                "Fill a searchable combobox / typeahead field that requires selecting from "
                "a dropdown (e.g. Greenhouse school, degree, discipline, country fields). "
                "Pass the element index and the desired value string. "
                "The action types a prefix, waits for suggestions, then clicks the best match. "
                "Use this INSTEAD of input+wait for any field with role=combobox or aria-autocomplete."
            )
        )
        async def fill_typeahead(
            index: int,
            value: str,
            browser_session: BrowserSession,
        ) -> ActionResult:
            JS = """
function(idx, val) {
    var map = window.__browserUse_selectorMap || window._bu_selectorMap;
    var el = null;
    if (map && map[idx]) { el = map[idx]; }
    if (!el) {
        var candidates = document.querySelectorAll(
            'input[role="combobox"], input[aria-autocomplete], input[aria-haspopup],' +
            '[role="combobox"] input, [contenteditable="true"]'
        );
        if (candidates[idx]) { el = candidates[idx]; }
    }
    if (!el) { return {ok: false, msg: 'element not found for index ' + idx}; }

    el.focus();
    el.value = '';
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));

    var prefix = val.length > 6 ? val.substring(0, 6) : val;
    el.value = prefix;
    el.dispatchEvent(new InputEvent('input', {bubbles: true, data: prefix, inputType: 'insertText'}));
    el.dispatchEvent(new KeyboardEvent('keydown', {bubbles: true, key: 'a'}));
    el.dispatchEvent(new KeyboardEvent('keyup', {bubbles: true}));

    return {ok: true, msg: 'typed prefix: ' + prefix};
}
"""
            JS_SELECT = """
function(val) {
    var lower = val.toLowerCase();
    var selectors = [
        '[role="option"]',
        '[role="listbox"] li',
        '[role="listbox"] [role="option"]',
        '[aria-selected]',
        '.select__option',
        '.greenhouse-option',
        'li[data-value]',
        'ul li'
    ];
    var opts = [];
    for (var i = 0; i < selectors.length; i++) {
        opts = Array.from(document.querySelectorAll(selectors[i]));
        if (opts.length > 0) break;
    }
    if (opts.length === 0) { return {ok: false, msg: 'no options found'}; }

    var match = opts.find(function(o) {
        return o.offsetParent !== null && o.textContent.toLowerCase().includes(lower);
    });
    if (!match) {
        match = opts.find(function(o) { return o.offsetParent !== null; });
    }
    if (!match) { return {ok: false, msg: 'no visible option found'}; }

    match.scrollIntoView({block: 'nearest'});
    match.click();
    match.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
    match.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
    return {ok: true, msg: 'clicked: ' + match.textContent.trim()};
}
"""
            try:
                cdp_session = await browser_session.get_or_create_cdp_session()

                async def _eval(code: str, *args):
                    # Build expression with baked-in args
                    expr = f"({code})({', '.join(json.dumps(a) for a in args)})"
                    expr = f"(function(){{ try{{ return {expr} }}catch(e){{return {{ok:false,msg:e.toString()}}}} }})()"
                    res = await cdp_session.cdp_client.send.Runtime.evaluate(
                        params={"expression": expr, "returnByValue": True, "awaitPromise": False},
                        session_id=cdp_session.session_id,
                    )
                    return (res.get("result") or {}).get("value") or {}

                # Type the prefix
                type_result = await _eval(JS, index, value)
                if not type_result.get("ok"):
                    return ActionResult(
                        error=f"fill_typeahead: could not type into field {index}: {type_result.get('msg')}",
                    )

                # Wait up to 3s for dropdown options to appear
                waited = 0.0
                select_result: dict = {}
                while waited < 3.0:
                    await asyncio.sleep(0.4)
                    waited += 0.4
                    select_result = await _eval(JS_SELECT, value)
                    if select_result.get("ok"):
                        break

                if select_result.get("ok"):
                    return ActionResult(
                        extracted_content=f"fill_typeahead: selected '{select_result.get('msg')}' for field {index}",
                        include_in_memory=True,
                        long_term_memory=f"Set typeahead field {index} to '{value}'",
                    )
                else:
                    return ActionResult(
                        error=(
                            f"fill_typeahead: typed prefix but could not select option for '{value}' "
                            f"(field {index}): {select_result.get('msg', 'no options appeared')}. "
                            "Try send_keys ArrowDown then Enter, or use a shorter search prefix."
                        ),
                    )

            except Exception as exc:
                return ActionResult(error=f"fill_typeahead error: {exc}")

        return tools

    except (ImportError, Exception) as e:
        print(f"[WARN] Could not register tools: {e}")
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
2. **Native `<select>` / listboxes:** open the control (click), then click the option whose visible text best matches the profile (or the closest allowed value).
3. **Searchable comboboxes & typeahead (e.g. Greenhouse school/country/degree):** call `fill_typeahead(index=<field_index>, value=<desired_value>)`. This action types a prefix, waits for the dropdown, and clicks the best matching option automatically — do NOT use `input` + `wait` loops for these fields. If `fill_typeahead` reports no options appeared, try `send_keys` with `ArrowDown` then `Enter` as a fallback.
4. **Radios & checkboxes:** click the option whose label matches the profile (use the element index for the control or its label).
5. For resume uploads, use `upload_file` with the exact path from the task.
6. For salary fields, use the Salary Range above. If unavailable, call search_missing_info.
7. For "Why here" fields, use the pre-synthesized answer above verbatim or lightly adapted.
8. For cover letter text fields, paste the Pre-Generated Cover Letter. If unavailable, write one from the profile and company data.
9. For cover letter file uploads, upload the PDF listed above. If unavailable, alert the user.
10. DO NOT open new browser tabs. Use search_missing_info for missing data.
11. For unforeseen behavioral or subjective questions, you MUST construct your response using only facts from the CANDIDATE PROFILE and the pre-researched context provided above (company mission, role description, recent news, etc.). Never invent information.
12. **Avoid stalling:** if you repeated the same wait/navigation goal without DOM change, change tactic (shorter typeahead text, click dropdown arrow, scroll suggestion list, or select by index).

## SAFETY RULES
- NEVER submit the application without user confirmation.
- STOP and alert if you detect a CAPTCHA, login requirement, or are uncertain about a field.
- Wait for pages to load fully before interacting; prefer **one** short wait after navigation, then concrete clicks/inputs instead of many identical waits.

You are applying as: {candidate_name}
"""


@dataclass
class SafeVLLMChat:
    """
    Implements browser-use 0.12.6's BaseChatModel protocol directly.
    Sends raw HTTP requests to a RunPod/vLLM endpoint, avoiding any OpenAI SDK
    or LangChain middleware.  Returns ``ChatInvokeCompletion`` exactly as
    browser-use expects.  Structured output (``output_format``) is handled by
    injecting the JSON schema into the system prompt and parsing the response.
    """

    model: str
    base_url: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 800

    # Cold-start retry settings
    _COLD_START_CODES: tuple = (502, 503, 504)
    _COLD_START_MAX_WAIT: int = 300
    _COLD_START_POLL: int = 15

    # ------------------------------------------------------------------ #
    #  Protocol properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        return self.model

    # ------------------------------------------------------------------ #
    #  Message / payload helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _msg_to_dict(msg) -> dict:
        """Convert any message object (browser-use, LangChain, or dict) to a raw dict."""
        if isinstance(msg, dict):
            return msg
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            # Multi-part content (images, text blocks) — flatten to text
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(p.get("text", str(p)))
                elif isinstance(p, str):
                    parts.append(p)
                else:
                    parts.append(getattr(p, "text", str(p)))
            content = "\n".join(parts)
        return {"role": str(role), "content": str(content)}

    def _build_payload(self, raw_messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": raw_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @property
    def _chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    @staticmethod
    def _is_cold_start(status_code: int, body: str) -> bool:
        if status_code in (502, 503, 504):
            return True
        lower = body.lower()
        return any(kw in lower for kw in ("loading", "starting", "initializing", "cold", "model not loaded"))

    # ------------------------------------------------------------------ #
    #  ainvoke — the only method browser-use 0.12.6 calls                  #
    # ------------------------------------------------------------------ #

    async def ainvoke(self, messages, output_format=None, **kwargs):
        """
        Send messages to the vLLM endpoint and return a ``ChatInvokeCompletion``.
        When *output_format* is a Pydantic model class, inject the JSON schema
        into the system prompt and parse the response into that model.
        """
        from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
        import httpx

        raw = [self._msg_to_dict(m) for m in messages]

        # When structured output is requested, tell the model to respond in JSON
        if output_format is not None:
            try:
                from browser_use.llm.schema import SchemaOptimizer
                schema = SchemaOptimizer.create_optimized_json_schema(output_format)
            except Exception:
                schema = output_format.model_json_schema()
            schema_instruction = (
                "\n\nYou MUST respond with ONLY a valid JSON object matching this schema. "
                "Do NOT include any text before or after the JSON.\n"
                f"<json_schema>\n{json.dumps(schema, indent=2)}\n</json_schema>"
            )
            if raw and raw[0]["role"] == "system":
                raw[0]["content"] += schema_instruction
            else:
                raw.insert(0, {"role": "system", "content": schema_instruction})

        payload = self._build_payload(raw)

        # Cold-start aware request loop
        waited = 0
        resp = None
        while True:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self._chat_url, headers=self._headers(), json=payload)
            if resp.status_code == 200:
                break
            body = ""
            try:
                body = resp.text
            except Exception:
                pass
            if self._is_cold_start(resp.status_code, body) and waited < self._COLD_START_MAX_WAIT:
                print(
                    f"[INFO] RunPod cold-start detected (HTTP {resp.status_code}), "
                    f"retrying in {self._COLD_START_POLL}s "
                    f"(waited {waited}s / {self._COLD_START_MAX_WAIT}s)..."
                )
                await asyncio.sleep(self._COLD_START_POLL)
                waited += self._COLD_START_POLL
                continue
            raise Exception(f"Error code: {resp.status_code} - {body[:500]}")

        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason", "stop")

        # Parse usage
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = ChatInvokeUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        # Structured output: parse JSON into the Pydantic model
        if output_format is not None:
            import re
            # Strip markdown fences if the model wrapped the JSON
            cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            try:
                parsed = output_format.model_validate_json(cleaned)
            except Exception:
                # Try to extract the first JSON object
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    parsed = output_format.model_validate_json(match.group())
                else:
                    raise
            return ChatInvokeCompletion(completion=parsed, usage=usage, stop_reason=finish)

        return ChatInvokeCompletion(completion=content, usage=usage, stop_reason=finish)


def _init_llm(model_name: Optional[str] = None) -> SafeVLLMChat:
    """
    Initialize the SafeVLLMChat client pointed at the remote cloud GPU vLLM endpoint.
    Reads RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY from environment variables.
    """
    endpoint_url = os.environ.get("RUNPOD_ENDPOINT_URL", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    model_name = model_name or os.environ.get(
        "RUNPOD_MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    )

    if not endpoint_url or not api_key:
        raise ConnectionError(
            "RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY must be set to connect to the cloud GPU. "
            "Configure them in the UI or set them as environment variables."
        )

    return SafeVLLMChat(
        model=model_name,
        base_url=endpoint_url,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
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

        _runpod_cm = managed_runpod_workers(log)
        await _runpod_cm.__aenter__()
        try:
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

            # Warm-up ping: send a tiny request to the endpoint so RunPod wakes the
            # GPU before the browser opens.  _agenerate already handles cold-start
            # 502/503/504 with its own retry loop, so this just gives us a log line
            # that confirms the model is live before we even open a browser window.
            log("Pinging GPU endpoint to ensure model is loaded (cold-start may take ~2 min)...", "info")
            try:
                from browser_use.llm.messages import UserMessage as BUUserMessage
                ping_msg = [BUUserMessage(content="ping")]
                await llm.ainvoke(ping_msg)
                log("GPU model is warm and responding", "success")
            except Exception as _ping_err:
                log(f"Warm-up ping warning (will retry per-step): {_ping_err}", "warning")

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

            # browser-use only allows upload_file paths listed in available_file_paths (exact match).
            resume_abs = str(Path(resume_path).expanduser().resolve())
            upload_paths: list[str] = [resume_abs]
            if cover_letter_pdf_full:
                cl_abs = str(Path(cover_letter_pdf_full).expanduser().resolve())
                if cl_abs not in upload_paths:
                    upload_paths.append(cl_abs)

            task = f"""
NAVIGATE TO: {job_url}

GOAL: Complete the job application form up to the review stage. DO NOT SUBMIT.

APPLYING FOR: {role_name} at {company_name}

## CANDIDATE PROFILE
{profile_content}

EXECUTION STEPS:
1. If a login is required, look for "Apply without account". If blocked, PAUSE and ask the user.
2. Upload the resume using upload_file with this exact path (copy verbatim): "{resume_abs}".
3. Fill all form fields using the CANDIDATE PROFILE above and the pre-researched data in the system prompt. For dropdowns and search-as-you-type fields, follow the FORM FILLING RULES (click option, or type-ahead then click suggestion or ArrowDown+Enter).
4. For cover letter file uploads: {"use upload_file with this exact path: " + repr(cover_letter_pdf_full) if cover_letter_pdf_full else "NO cover letter file is available — SKIP the cover letter upload entirely. Do NOT guess a filename."}.
5. Click "Next" / "Continue" until you reach the review page.
6. HALT immediately. Notify the user: "Application ready for review."
"""

            # Run agent with browser crash retry loop
            browser_retry_count = 0
            result = None

            while browser_retry_count <= config.browser_restart_retries:
                try:
                    browser_session = BrowserSession(
                        browser_profile=BrowserProfile(
                            headless=False,
                            # Give the Greenhouse SPA 2s minimum + up to 5s network idle
                            # before handing the DOM to the agent — eliminates blank-page loops.
                            minimum_wait_page_load_time=2.0,
                            wait_for_network_idle_page_load_time=5.0,
                        )
                    )

                    agent_kwargs: dict = dict(
                        task=task,
                        llm=llm,  # type: ignore[arg-type]
                        browser_session=browser_session,
                        extend_system_message=system_prompt,
                        include_attributes=[
                            "title",
                            "type",
                            "name",
                            "role",
                            "id",
                            "aria-label",
                            "aria-expanded",
                            "aria-autocomplete",
                            "aria-controls",
                            "placeholder",
                            "value",
                            "data-testid",
                        ],
                        llm_timeout=300,
                        available_file_paths=upload_paths,
                        # Long job forms: give planning a bit more room before nudges; helps typeahead-heavy pages.
                        planning_exploration_limit=8,
                        planning_replan_on_stall=5,
                        use_thinking=_env_flag("BROWSER_USE_USE_THINKING", False),
                        # browser-use's auto-URL-extraction breaks when the task contains multiple
                        # URLs (job URL + LinkedIn + GitHub) — it finds >1 and skips navigation,
                        # leaving the browser on about:blank for 5+ wasted wait steps.
                        # We handle navigation ourselves via initial_actions instead.
                        directly_open_url=False,
                        initial_actions=[{"navigate": {"url": job_url, "new_tab": False}}],
                    )
                    if search_tools is not None:
                        agent_kwargs["tools"] = search_tools

                    agent = Agent(**agent_kwargs)

                    log(f"Navigating to: {job_url}", "action")
                    state.update_phase(AgentPhase.FILLING_FORM, f"Navigating to {job_url}")
                    state_manager.save_state(state)

                    result = await agent.run()

                    # CAPTCHA detection — only check the agent's final text output,
                    # NOT str(result) which includes DOM snapshots that always
                    # contain invisible reCAPTCHA badge references.
                    _final_text = (result.final_result() or "") if result else ""
                    result_str = _final_text.lower()
                    if re.search(r"(?i)(?<!re)captcha", result_str):
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

            # Basic validation: check that the result looks like real progress.
            # If all_model_outputs is empty the LLM never produced any action —
            # treat that as a failure (e.g. RunPod cold-start killed every step).
            result_str = str(result).lower() if result else ""
            model_outputs = getattr(result, "all_model_outputs", None)
            if model_outputs is not None and len(model_outputs) == 0:
                log(
                    "Agent produced zero model outputs — GPU may not have been ready. "
                    "Re-run when the endpoint is warm.",
                    "warning",
                )
                if application_id:
                    db.update_status(application_id, ApplicationStatus.FAILED,
                                     notes="Zero model outputs — RunPod cold-start suspected")
                if state:
                    state_manager.mark_failed(state, "zero model outputs")
                return {
                    "success": False,
                    "message": "Agent produced zero model outputs. The GPU endpoint may still be cold-starting — please retry in 1-2 minutes.",
                    "application_id": application_id,
                    "session_id": state.session_id if state else None,
                    "can_resume": True,
                }

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

        finally:
            await _runpod_cm.__aexit__(None, None, None)

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
        import traceback
        traceback.print_exc()
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

    # Force UTF-8 stdout/stderr so Windows cp1252 consoles don't choke on emoji
    # coming from browser-use's internal logger.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    def print_usage():
        print("\nUsage:")
        print("  python apply_agent.py <job_url> <resume_path> [--force]  - Run agent for a job")
        print("  python apply_agent.py resume <session_id>                - Resume a session")
        print("  python apply_agent.py sessions                           - List recoverable sessions")
        print("  python apply_agent.py check                              - Check for active session")
        print("\nFlags:")
        print("  --force   Skip duplicate URL check (useful when re-testing the same job)")

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
        force_run = "--force" in sys.argv
        profile_path = Path(__file__).parent / "my_profile.md"

        print(f"Starting agent for: {job_url}")
        print(f"Using profile: {profile_path}")
        print(f"Using resume: {resume_path}")
        if force_run:
            print("[INFO] --force flag set: skipping duplicate URL check")

        result = asyncio.run(
            run_agent(
                job_url=job_url,
                resume_path=resume_path,
                profile_path=str(profile_path),
                skip_duplicate_check=force_run,
            )
        )

        print(f"\nResult: {result}")

        if result.get("can_resume") and result.get("session_id"):
            print(f"\nTo resume: python apply_agent.py resume {result['session_id']}")

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
