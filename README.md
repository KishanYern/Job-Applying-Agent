# AI Job Application Agent V2

A two-phase, cloud-optimized autonomous job application agent. Phase 1 pre-researches every company and pre-generates all application materials using a serverless 70B model. Phase 2 runs the browser automation against a remote 32B coding model on a rented cloud GPU, consuming only pre-cached data.

## Architecture

### Phase 1 — Discovery & Research (local trigger, cloud inference)

Runs on your machine as a background task before any applications are submitted.

- **Trigger:** "Search Jobs" or "Run Discovery" button in the UI (or a scheduled cron job via `job_scraper.py`)
- **Model:** Llama-3.3-70B-Instruct via [Groq](https://console.groq.com) serverless API
- **Search:** [Tavily](https://app.tavily.com) for company intel and salary data
- **Output:** For each new job, the pipeline caches to SQLite:
  - Company values, mission, products, recent news, competitors
  - Tech stack, required skills, salary range
  - Pre-generated cover letter
  - Pre-synthesized "Why do you want to work here?" answer

### Phase 2 — Autonomous Application (cloud GPU)

Runs on a rented cloud GPU instance (RunPod / Vast.ai).

- **Model:** Qwen2.5-Coder-32B at 8-bit quantization via OpenAI-compatible vLLM endpoint
- **Browser:** Playwright (headless=False, visible to the user)
- **Agent:** browser-use with DOM pruning enabled
- **Data flow:** Loads all pre-researched data from SQLite → injects into system prompt → fills forms without opening new tabs
- **Fallback:** On-the-spot Tavily tool call if cached data is missing (no browser tab required)

```
Phase 1 (local)                    Phase 2 (cloud GPU)
─────────────────────────────      ────────────────────────────────
job_scraper.py                     apply_agent.py
  │                                  │
  ├─► Tavily search                  ├─► Load profile + DB cache
  ├─► Groq Llama-3.3-70B            ├─► Build system prompt
  │     (extract JSON)               ├─► Qwen2.5-Coder-32B (remote)
  └─► SQLite DB ──────────────────►  └─► Playwright browser
        companies                          └─► Fill forms
        job_requirements                   └─► Tavily fallback tool
        saved_answers
```

## Prerequisites

- Python 3.10+
- API Keys:
  - **Groq** — free tier at [console.groq.com](https://console.groq.com)
  - **Tavily** — free tier at [app.tavily.com](https://app.tavily.com)
  - **RunPod or Vast.ai** — rent an RTX A6000 (48GB VRAM) or dual RTX 4090 instance running a vLLM server with `qwen2.5-coder-32b-instruct`

No local GPU required. No Ollama installation required.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/job-applying-agent.git
   cd job-applying-agent
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   playwright install
   ```

3. **Set up your profile**

   Create `my_profile.md` in the project root with your name, contact info, education, work experience, skills, and standard application answers.

4. **Add your resume**

   Place your resume PDF in the project root (e.g., `YourName_Resume.pdf`). Update the path in the UI.

5. **Configure API keys**

   Either set environment variables:

   ```bash
   set GROQ_API_KEY=gsk_...
   set TAVILY_API_KEY=tvly-...
   set RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/<id>/openai/v1
   set RUNPOD_API_KEY=...
   ```

   Or enter them directly in the sidebar of the Streamlit UI — they are persisted for the session.

## Usage

```bash
streamlit run app.py
```

### Recommended workflow

1. **Discover tab** — Click "Search Jobs" (optionally check "Run Phase 1 Research" to research + pre-generate cover letters in one step).
2. **Auto-Apply tab** — Click "Run Discovery Now" if jobs are missing Phase 1 data. Then click "Start Auto-Apply" to begin Phase 2.
3. **Review** — The agent halts before submitting. Review each application in the browser, then submit manually.

## Cost Model

| Item | Cost |
|------|------|
| Phase 1: research 1 company + generate cover letter (~2,500 tokens) | ~$0.0017 |
| Phase 2: fill 1 application (3 min compute on RTX A6000 @ $0.37-$0.49/hr) | ~$0.019-$0.025 |
| **Total per application** | **~$0.02-$0.026** |

On-the-spot Tavily fallback queries add ~$0.0005 each.

## File Overview

| File | Role |
|------|------|
| `app.py` | Streamlit UI — API key config, discover tab, auto-apply tab |
| `job_scraper.py` | Phase 1: scrapes GitHub job boards, runs research pipeline |
| `cover_letter.py` | Phase 1: generates cover letters via Groq |
| `apply_agent.py` | Phase 2: browser-use agent with pre-loaded data |
| `application_db.py` | SQLite schema — applications, companies, job_requirements, saved_answers |
| `agent_state.py` | Session persistence and crash recovery |
| `notifications.py` | Desktop alerts for CAPTCHA, errors, daily summaries |
| `my_profile.md` | Your profile (not committed — add manually) |
