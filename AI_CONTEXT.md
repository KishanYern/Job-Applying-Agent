Project Context: Cloud-Optimized Autonomous Job Application Agent V2

1. Project Goal

We are building a two-phase, cloud-optimized autonomous job application agent.

The Problem: Applying to jobs is tedious. A single application typically requires 10–20 minutes of manual form-filling, company research, and cover letter writing.

The Solution: A two-phase AI pipeline. Phase 1 pre-researches every company and pre-generates all application materials in the background using a serverless 70B model. Phase 2 runs the browser automation against a remote 32B coding model on a rented cloud GPU, filling forms using only pre-cached data — no live research, no multi-tab browsing, no local GPU required.

Key Constraint: Personal data (resume, phone number, address) must stay local. It is written into a local SQLite DB and a local profile file and never sent to a third-party storage service. External APIs (Groq, Tavily, RunPod) only receive job descriptions and company names — not the user's personal details.

2. The Tech Stack

Language: Python 3.10+

Frontend: Streamlit

Phase 1 Inference: Groq API (Llama-3.3-70B-Instruct) via langchain-groq
Phase 1 Search: Tavily API (TavilyClient) for company research and salary data
Phase 2 Inference: Qwen2.5-Coder-32B on RunPod/Vast.ai via ChatOpenAI (OpenAI-compatible vLLM endpoint)

Browser Automation: Playwright (headless=False so the user can watch)
Agent Framework: browser-use with DOM pruning (include_attributes configured)
Data Handling: langchain-openai for Phase 2 LLM interface; langchain-groq for Phase 1
Database: SQLite via application_db.py (local, never uploaded anywhere)

3. Architecture Overview

A. Phase 1 — Discovery & Research Pipeline (job_scraper.py + cover_letter.py)

Triggered from the UI Discover tab or run as a cron job.

Steps for each new job URL:
1. research_company(job, http_client): Tavily searches for company mission/values/products/salary, fetches job description text, sends both to Groq Llama-3.3-70B for structured JSON extraction.
2. cache_company_info(): Upserts extracted data into the `companies` table (name, about_summary, values_summary, competitors, recent_news).
3. add_application(): Registers the job in the `applications` table, returns job_id.
4. generate_cover_letter_async(): Calls Groq to write a tailored cover letter using the extracted company data + candidate profile.
5. save_job_requirements(): Stores tech_stack, skills_required, salary_range, cover_letter_text in the `job_requirements` table linked to job_id.
6. save_answer(): Caches the pre-synthesized "why here" answer in `saved_answers`.

B. Phase 2 — Autonomous Application Pipeline (apply_agent.py)

Triggered from the Auto-Apply tab or Manual tab. Requires the cloud GPU endpoint to be running.

Steps:
1. build_system_prompt(): Queries SQLite for company_info, job_requirements, saved_answers. Injects all pre-researched data directly into the system prompt — no live research needed.
2. _init_llm(): Instantiates ChatOpenAI pointed at the RunPod/Vast.ai vLLM endpoint serving Qwen2.5-Coder-32B.
3. Agent(task, llm, extend_system_message, include_attributes, browser_context_kwargs): Launches browser-use with DOM pruning flags. The agent fills forms using cached data.
4. search_missing_info tool: If data is absent from the cache, the agent calls this LangChain tool to query Tavily directly — never opens a new browser tab.
5. CAPTCHA/login: Agent halts and notifies user.

C. The Frontend (app.py)

Sidebar: API key configuration (Groq, Tavily, RunPod endpoint + key). Keys are stored in st.session_state and pushed to os.environ before each agent call.

Discover Tab: Scrapes GitHub job boards. Optional "Run Phase 1 Research" checkbox triggers process_discovered_job() for each new listing.

Auto-Apply Tab: Shows Phase 1 readiness for each queued job. "Run Discovery Now" button researches unresearched jobs. "Start Auto-Apply" launches Phase 2 sequentially.

Manual Tab: Single-URL Phase 2 application.

History Tab: Track all applications, update statuses (interview, offer, rejected).

4. Database Schema (application_db.py)

Tables:
- applications: id, company, role, job_url (UNIQUE), location, status, source, resume_used, notes, created_at, updated_at, applied_at
- saved_answers: id, question_pattern, answer, company, created_at, used_count
- companies: id, name (UNIQUE), website, about_summary, values_summary, salary_data, competitors, recent_news, last_researched, created_at
- job_requirements: id, job_id (FK → applications.id), tech_stack, skills_required, salary_range, cover_letter_text, created_at

Dataclasses: Application, SavedAnswer, CompanyInfo, JobRequirement

Key methods:
- cache_company_info(name, about_summary, values_summary, competitors, recent_news, ...)
- save_job_requirements(job_id, tech_stack, skills_required, salary_range, cover_letter_text)
- get_job_requirements(job_id) → Optional[JobRequirement]
- get_company_info(name) → Optional[CompanyInfo]
- find_answer(question, company) → Optional[SavedAnswer]

5. Specific File Requirements

apply_agent.py
- Exports: async def run_agent(job_url, resume_path, profile_path, log_callback, company, role, job_id, skip_duplicate_check, retry_config, resume_session_id)
- No model_name parameter — model is fixed to Qwen2.5-Coder-32B via RUNPOD_ENDPOINT_URL + RUNPOD_API_KEY env vars
- build_system_prompt(profile_content, company, role, job_id) queries DB for all pre-researched data

cover_letter.py
- Uses ChatGroq(model="llama-3.3-70b-versatile") — no model_name parameter
- Exports: generate_cover_letter_async(profile_content, job), generate_cover_letter_sync(profile_content, job)
- No get_cover_letter_instructions() — cover letters are never generated inline by the browser agent

job_scraper.py
- Exports: research_company(job, http_client), process_discovered_job(job, profile_content, http_client)
- get_job_urls_sync(keywords, locations, job_type, run_research, profile_content) — run_research=True triggers Phase 1

6. Environment Variables Required

GROQ_API_KEY        - Groq API key for Phase 1 Llama-3.3-70B inference
TAVILY_API_KEY      - Tavily API key for company research and fallback search
RUNPOD_ENDPOINT_URL - vLLM endpoint URL for Qwen2.5-Coder-32B on cloud GPU
RUNPOD_API_KEY      - API key for the cloud GPU provider

7. Development Guidelines

Keep it Modular: Phase 1 (job_scraper.py, cover_letter.py) and Phase 2 (apply_agent.py) are completely decoupled.

Phase Ordering: Phase 1 must complete for a job before Phase 2 runs. The UI enforces this with a readiness check and a "Run Discovery Now" button.

Error Handling: If the cloud GPU endpoint is unavailable, fail gracefully with a clear "RUNPOD_ENDPOINT_URL not configured" message. If Groq/Tavily fail during Phase 1, log a warning and continue — Phase 2 will use the search_missing_info fallback tool.

Privacy: Personal data from my_profile.md is only injected locally into the system prompt. It is never sent to Tavily or Groq — only company names and job descriptions are sent to external APIs.
