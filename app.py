"""
Local AI Job Application Agent - Streamlit UI
A privacy-focused, automated job application tool running entirely on local hardware.
"""

import os
import streamlit as st
import asyncio
import time
from pathlib import Path
from datetime import datetime

# Enable nested async event loops (required for Streamlit + async browser-use)
import nest_asyncio
nest_asyncio.apply()

# Local imports
from application_db import get_db, ApplicationStatus
from job_scraper import get_job_urls_sync, JobListing

# Notifications and state management
from notifications import (
    start_daily_summary_scheduler,
    stop_daily_summary_scheduler,
    get_scheduler,
    notify_captcha,
    send_daily_summary,
    NotificationConfig,
    configure_notifications,
)
from agent_state import get_state_manager, AgentPhase

# Page configuration
st.set_page_config(
    page_title="AI Job Application Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .job-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .queue-count {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A5F;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2D4A6F;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "agent_running" not in st.session_state:
        st.session_state.agent_running = False
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = "idle"
    if "scraped_jobs" not in st.session_state:
        st.session_state.scraped_jobs = []
    if "auto_apply_running" not in st.session_state:
        st.session_state.auto_apply_running = False
    if "current_job_index" not in st.session_state:
        st.session_state.current_job_index = 0
    if "auto_apply_paused" not in st.session_state:
        st.session_state.auto_apply_paused = False
    # Notification settings
    if "notifications_enabled" not in st.session_state:
        st.session_state.notifications_enabled = True
    if "sound_enabled" not in st.session_state:
        st.session_state.sound_enabled = True
    if "daily_summary_time" not in st.session_state:
        st.session_state.daily_summary_time = "18:00"
    if "daily_scheduler_started" not in st.session_state:
        st.session_state.daily_scheduler_started = False
    # Session recovery
    if "pending_resume_session" not in st.session_state:
        st.session_state.pending_resume_session = None
    # V2 API keys (loaded from env if present)
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if "tavily_api_key" not in st.session_state:
        st.session_state.tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if "runpod_endpoint_url" not in st.session_state:
        st.session_state.runpod_endpoint_url = os.environ.get("RUNPOD_ENDPOINT_URL", "")
    if "runpod_api_key" not in st.session_state:
        st.session_state.runpod_api_key = os.environ.get("RUNPOD_API_KEY", "")


def start_notification_scheduler():
    """Start the daily summary scheduler if enabled."""
    if st.session_state.notifications_enabled and not st.session_state.daily_scheduler_started:
        db = get_db()
        start_daily_summary_scheduler(db, st.session_state.daily_summary_time)
        st.session_state.daily_scheduler_started = True


def update_notification_config():
    """Update notification configuration from session state."""
    config = NotificationConfig(
        enable_desktop=st.session_state.notifications_enabled,
        enable_sound=st.session_state.sound_enabled,
        daily_summary_time=st.session_state.daily_summary_time,
    )
    configure_notifications(config)


def load_profile() -> str:
    """Load the user profile from my_profile.md."""
    profile_path = Path(__file__).parent / "my_profile.md"
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8")
    return ""


def get_resume_path() -> str | None:
    """Get the default resume path."""
    repo_resume_path = Path(__file__).parent / "Kishan_Yerneni_Resume.pdf"
    if repo_resume_path.exists():
        return str(repo_resume_path)
    return None


def add_log(message: str, level: str = "info"):
    """Add a log message to the session state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "action": "🤖"}.get(level, "ℹ️")
    st.session_state.agent_logs.append(f"[{timestamp}] {icon} {message}")


def _sync_api_keys_to_env():
    """Push API keys from session state into environment variables for sub-modules."""
    if st.session_state.get("groq_api_key"):
        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
    if st.session_state.get("tavily_api_key"):
        os.environ["TAVILY_API_KEY"] = st.session_state.tavily_api_key
    if st.session_state.get("runpod_endpoint_url"):
        os.environ["RUNPOD_ENDPOINT_URL"] = st.session_state.runpod_endpoint_url
    if st.session_state.get("runpod_api_key"):
        os.environ["RUNPOD_API_KEY"] = st.session_state.runpod_api_key


async def run_agent_for_job(
    job_url: str,
    resume_path: str,
    company: str = None,
    role: str = None,
    job_id: int = None,
    skip_duplicate: bool = False,
    resume_session_id: str = None,
) -> dict:
    """Run the Phase 2 job application agent for a single job."""
    try:
        _sync_api_keys_to_env()
        from apply_agent import run_agent, resume_agent

        if resume_session_id:
            result = await resume_agent(
                session_id=resume_session_id,
                log_callback=add_log,
            )
        else:
            result = await run_agent(
                job_url=job_url,
                resume_path=resume_path,
                profile_path=str(Path(__file__).parent / "my_profile.md"),
                log_callback=add_log,
                company=company,
                role=role,
                job_id=job_id,
                skip_duplicate_check=skip_duplicate,
            )
        return result
    except Exception as e:
        return {"success": False, "message": str(e), "can_resume": False}


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ── API Keys ──────────────────────────────────────────────────────────
        st.subheader("🔑 API Keys")
        st.caption("Phase 1 (Discovery) uses Groq + Tavily. Phase 2 (Apply) uses your cloud GPU.")

        groq_key = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            help="Get a free key at console.groq.com — used for Llama-3.3-70B Phase 1 research",
        )
        if groq_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = groq_key
            os.environ["GROQ_API_KEY"] = groq_key

        tavily_key = st.text_input(
            "Tavily API Key",
            value=st.session_state.tavily_api_key,
            type="password",
            help="Get a key at app.tavily.com — used for company research and fallback search",
        )
        if tavily_key != st.session_state.tavily_api_key:
            st.session_state.tavily_api_key = tavily_key
            os.environ["TAVILY_API_KEY"] = tavily_key

        gpu_endpoint = st.text_input(
            "Cloud GPU Endpoint URL",
            value=st.session_state.runpod_endpoint_url,
            placeholder="https://api.runpod.ai/v2/<id>/openai/v1",
            help="RunPod or Vast.ai vLLM endpoint hosting Qwen2.5-Coder-32B",
        )
        if gpu_endpoint != st.session_state.runpod_endpoint_url:
            st.session_state.runpod_endpoint_url = gpu_endpoint
            os.environ["RUNPOD_ENDPOINT_URL"] = gpu_endpoint

        gpu_key = st.text_input(
            "Cloud GPU API Key",
            value=st.session_state.runpod_api_key,
            type="password",
            help="RunPod / Vast.ai API key for Qwen2.5-Coder-32B Phase 2 inference",
        )
        if gpu_key != st.session_state.runpod_api_key:
            st.session_state.runpod_api_key = gpu_key
            os.environ["RUNPOD_API_KEY"] = gpu_key

        # API key status indicators
        api_ready = all([
            st.session_state.groq_api_key,
            st.session_state.tavily_api_key,
            st.session_state.runpod_endpoint_url,
            st.session_state.runpod_api_key,
        ])
        if api_ready:
            st.success("✓ All API keys configured")
        else:
            missing = [
                name for name, val in [
                    ("Groq", st.session_state.groq_api_key),
                    ("Tavily", st.session_state.tavily_api_key),
                    ("GPU Endpoint", st.session_state.runpod_endpoint_url),
                    ("GPU Key", st.session_state.runpod_api_key),
                ]
                if not val
            ]
            st.warning(f"Missing: {', '.join(missing)}")

        st.divider()

        # ── Active Model Info ─────────────────────────────────────────────────
        st.subheader("🧠 Active Models")
        phase1_ready = bool(st.session_state.groq_api_key and st.session_state.tavily_api_key)
        phase2_ready = bool(st.session_state.runpod_endpoint_url and st.session_state.runpod_api_key)

        p1_icon = "✅" if phase1_ready else "⚠️"
        p2_icon = "✅" if phase2_ready else "⚠️"

        st.markdown(
            f"{p1_icon} **Phase 1** — Groq / `llama-3.3-70b-versatile`\n\n"
            f"{p2_icon} **Phase 2** — `qwen2.5-coder-32b-instruct` (cloud GPU)"
        )
        if not phase1_ready:
            st.caption("Set Groq + Tavily keys to enable Phase 1 research.")
        if not phase2_ready:
            st.caption("Set GPU Endpoint + Key to enable Phase 2 applications.")

        st.divider()
        
        # Notification Settings
        st.header("🔔 Notifications")
        
        notifications_enabled = st.checkbox(
            "Desktop Notifications",
            value=st.session_state.notifications_enabled,
            help="Show desktop alerts for CAPTCHAs and errors"
        )
        if notifications_enabled != st.session_state.notifications_enabled:
            st.session_state.notifications_enabled = notifications_enabled
            update_notification_config()
        
        sound_enabled = st.checkbox(
            "Sound Alerts",
            value=st.session_state.sound_enabled,
            help="Play sounds for important events"
        )
        if sound_enabled != st.session_state.sound_enabled:
            st.session_state.sound_enabled = sound_enabled
            update_notification_config()
        
        daily_time = st.text_input(
            "Daily Summary Time (HH:MM)",
            value=st.session_state.daily_summary_time,
            help="Time to send daily application summary",
            max_chars=5
        )
        if daily_time != st.session_state.daily_summary_time:
            # Validate time format
            try:
                h, m = map(int, daily_time.split(":"))
                if 0 <= h <= 23 and 0 <= m <= 59:
                    st.session_state.daily_summary_time = daily_time
                    update_notification_config()
                    # Restart scheduler with new time
                    if st.session_state.daily_scheduler_started:
                        stop_daily_summary_scheduler()
                        st.session_state.daily_scheduler_started = False
                        start_notification_scheduler()
            except ValueError:
                pass
        
        col_notif1, col_notif2 = st.columns(2)
        with col_notif1:
            if st.button("📊 Summary Now", help="Send daily summary now"):
                db = get_db()
                send_daily_summary(db)
                st.success("Summary sent!")
        with col_notif2:
            if st.button("🔔 Test Alert", help="Test notification"):
                from notifications import notify, NotificationType
                notify("Test Alert", "Notifications are working!", NotificationType.INFO)
                st.success("Alert sent!")
        
        st.divider()
        
        # Session Recovery
        state_manager = get_state_manager()
        recoverable = state_manager.get_recoverable_sessions(max_age_hours=24)
        active_session = state_manager.get_active_session()
        
        if active_session or recoverable:
            st.header("🔄 Recovery")
            
            if active_session:
                st.warning(f"⚠️ Interrupted session found!")
                st.markdown(f"**{active_session.role}** at **{active_session.company}**")
                st.caption(f"Phase: {active_session.phase.value}")
                
                if st.button("🔄 Resume Session", use_container_width=True):
                    st.session_state.pending_resume_session = active_session.session_id
                    st.rerun()
                
                if st.button("🗑️ Dismiss", use_container_width=True):
                    state_manager.mark_failed(active_session, "Dismissed by user")
                    st.rerun()
            
            elif recoverable:
                with st.expander(f"📋 {len(recoverable)} recoverable session(s)"):
                    for sess in recoverable[:5]:
                        st.markdown(f"**{sess.company}** - {sess.role}")
                        st.caption(f"{sess.phase.value} | {sess.updated_at[:16]}")
                        if st.button("Resume", key=f"resume_{sess.session_id}"):
                            st.session_state.pending_resume_session = sess.session_id
                            st.rerun()
            
            st.divider()
        
        # Profile Preview
        st.header("👤 Your Profile")
        profile_content = load_profile()
        if profile_content:
            with st.expander("View Profile Data", expanded=False):
                st.text(profile_content[:2000] + "..." if len(profile_content) > 2000 else profile_content)
            st.success("✓ Profile loaded")
        else:
            st.error("⚠️ my_profile.md not found!")
        
        # Resume status
        resume_path = get_resume_path()
        if resume_path:
            st.success("✓ Resume found")
        else:
            st.error("⚠️ Resume not found!")
        
        st.divider()
        
        # Application Statistics
        st.header("📊 Stats")
        db = get_db()
        stats = db.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats["total"])
        with col2:
            st.metric("This Week", stats["last_7_days"])
        
        # Queue count
        queued = stats.get("by_status", {}).get("queued", 0)
        if queued > 0:
            st.info(f"📋 **{queued}** jobs in queue")
        
        st.divider()
        
        # Quick links
        st.header("📖 Mode Guide")
        st.markdown("""
        - **🔍 Discover**: Find & research jobs (Phase 1)
        - **🚀 Auto-Apply**: Batch apply to queue (Phase 2)
        - **📝 Manual**: Apply to single URL (Phase 2)
        - **📚 History**: Track applications
        """)


def render_discover_tab():
    """Render the job discovery tab."""
    st.header("🔍 Discover Jobs")
    st.markdown("Find jobs from curated GitHub repositories (SimplifyJobs, Ouckah, pittcsc)")
    
    # Search filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_type = st.selectbox(
            "Job Type",
            options=["internship", "new-grad", "all"],
            index=0
        )
    
    with col2:
        location_filter = st.text_input(
            "Location Filter",
            placeholder="e.g., Remote, Houston, CA",
            help="Comma-separated locations (leave empty for all)"
        )
    
    with col3:
        keyword_filter = st.text_input(
            "Role Keywords",
            placeholder="e.g., machine learning, backend",
            help="Comma-separated keywords (leave empty for default SWE/AI/DS)"
        )
    
    # Research toggle
    run_phase1 = st.checkbox(
        "Run Phase 1 Research after discovering",
        value=False,
        help="Automatically research each new job via Groq + Tavily and pre-generate cover letters. "
             "Requires Groq and Tavily API keys to be configured.",
    )

    # Search button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        search_clicked = st.button("🔍 Search Jobs", use_container_width=True)

    if search_clicked:
        _sync_api_keys_to_env()
        profile_content = load_profile() if run_phase1 else ""
        if run_phase1 and not profile_content:
            st.warning("my_profile.md not found — Phase 1 research skipped.")
            run_phase1 = False

        spinner_msg = (
            "Fetching jobs and running Phase 1 research (this may take a few minutes)..."
            if run_phase1
            else "Fetching jobs from GitHub repositories..."
        )
        with st.spinner(spinner_msg):
            locations = [loc.strip() for loc in location_filter.split(",")] if location_filter else None
            keywords = [kw.strip() for kw in keyword_filter.split(",")] if keyword_filter else None

            jobs = get_job_urls_sync(
                keywords=keywords,
                locations=locations,
                job_type=job_type if job_type != "all" else None,
                run_research=run_phase1,
                profile_content=profile_content,
            )
            st.session_state.scraped_jobs = jobs
    
    # Display results
    jobs = st.session_state.scraped_jobs
    
    if jobs:
        st.success(f"Found **{len(jobs)}** matching jobs!")
        
        # Queue controls
        st.markdown("---")
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            num_to_queue = st.number_input("Jobs to queue", min_value=1, max_value=min(100, len(jobs)), value=min(10, len(jobs)))
        
        with col_q2:
            st.write("")  # Spacing
            st.write("")
            if st.button(f"➕ Add {num_to_queue} to Queue", use_container_width=True):
                db = get_db()
                added = 0
                skipped = 0
                for job in jobs[:num_to_queue]:
                    # Check if already in database
                    if not db.is_duplicate(job.apply_url):
                        db.add_application(
                            company=job.company,
                            role=job.role,
                            job_url=job.apply_url,
                            location=job.location,
                            status=ApplicationStatus.QUEUED,
                            source=job.source_repo
                        )
                        added += 1
                    else:
                        skipped += 1
                st.success(f"✅ Added {added} jobs to queue ({skipped} duplicates skipped)")
                st.rerun()
        
        with col_q3:
            st.write("")
            st.write("")
            if st.button("➕ Add ALL to Queue", use_container_width=True):
                db = get_db()
                added = 0
                skipped = 0
                for job in jobs:
                    if not db.is_duplicate(job.apply_url):
                        db.add_application(
                            company=job.company,
                            role=job.role,
                            job_url=job.apply_url,
                            location=job.location,
                            status=ApplicationStatus.QUEUED,
                            source=job.source_repo
                        )
                        added += 1
                    else:
                        skipped += 1
                st.success(f"✅ Added {added} jobs to queue ({skipped} duplicates skipped)")
                st.rerun()
        
        # Job listings
        st.markdown("---")
        st.subheader(f"📋 Job Listings ({len(jobs)})")
        
        for i, job in enumerate(jobs[:50]):  # Show first 50
            db = get_db()
            is_duplicate = db.is_duplicate(job.apply_url)
            
            with st.expander(
                f"{'✅' if is_duplicate else '🆕'} {job.company} - {job.role}",
                expanded=False
            ):
                st.markdown(f"**Location:** {job.location}")
                st.markdown(f"**Source:** {job.source_repo}")
                st.markdown(f"**URL:** [{job.apply_url[:60]}...]({job.apply_url})")
                
                if is_duplicate:
                    st.info("Already in database")
                else:
                    if st.button(f"➕ Add to Queue", key=f"add_{i}"):
                        db.add_application(
                            company=job.company,
                            role=job.role,
                            job_url=job.apply_url,
                            location=job.location,
                            status=ApplicationStatus.QUEUED,
                            source=job.source_repo
                        )
                        st.success("Added to queue!")
                        st.rerun()
        
        if len(jobs) > 50:
            st.info(f"Showing 50 of {len(jobs)} jobs. Add to queue to process more.")
    else:
        st.info("Click 'Search Jobs' to discover opportunities from GitHub job boards.")


def render_auto_apply_tab():
    """Render the auto-apply tab (Phase 2)."""
    st.header("🚀 Auto-Apply Mode")
    st.markdown("Phase 2: Autonomously fills application forms using pre-researched data. You only intervene for CAPTCHAs.")

    db = get_db()
    queued_jobs = db.get_all_applications(status=ApplicationStatus.QUEUED, limit=500)
    resume_path = get_resume_path()

    # Status overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 In Queue", len(queued_jobs))
    with col2:
        completed = len(db.get_all_applications(status=ApplicationStatus.COMPLETED, limit=1000))
        st.metric("✅ Completed", completed)
    with col3:
        failed = len(db.get_all_applications(status=ApplicationStatus.FAILED, limit=1000))
        st.metric("❌ Failed", failed)

    st.markdown("---")

    if not resume_path:
        st.error("❌ No resume found! Please add your resume PDF to the project folder.")
        return

    if not queued_jobs:
        st.info("📭 No jobs in queue. Go to **Discover** tab to find and queue jobs.")
        return

    # Phase 1 readiness check
    jobs_missing_research = [
        job for job in queued_jobs
        if db.get_job_requirements(job.id) is None
    ]

    if jobs_missing_research:
        st.warning(
            f"⚠️ **{len(jobs_missing_research)} of {len(queued_jobs)} queued jobs have not been researched** "
            f"(no Phase 1 data cached). Running Phase 2 without research means the agent will have "
            f"no pre-generated cover letters, salary data, or 'why here' answers."
        )
        col_disc1, col_disc2 = st.columns([1, 3])
        with col_disc1:
            if st.button("🔬 Run Discovery Now", use_container_width=True, type="secondary"):
                _sync_api_keys_to_env()
                profile_content = load_profile()
                if not profile_content:
                    st.error("my_profile.md not found — cannot run research.")
                elif not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
                    st.error("Groq and Tavily API keys must be configured in the sidebar.")
                else:
                    import httpx as _httpx
                    import threading as _threading
                    from job_scraper import process_discovered_job, JobListing as _JobListing
                    progress = st.progress(0)
                    total = len(jobs_missing_research)
                    # Semaphore limits concurrent Groq + Tavily API calls:
                    # 5 jobs × 3 calls each = up to 15 simultaneous requests,
                    # safely within free-tier rate limits for both services.
                    _CONCURRENCY = 5
                    with st.spinner(f"Running Phase 1 research for {total} jobs (up to {_CONCURRENCY} concurrent)..."):
                        async def _run_research():
                            semaphore = asyncio.Semaphore(_CONCURRENCY)
                            completed = 0
                            lock = _threading.Lock()

                            async def _research_one(job_app):
                                nonlocal completed
                                listing = _JobListing(
                                    company=job_app.company,
                                    role=job_app.role,
                                    location=job_app.location or "",
                                    apply_url=job_app.job_url,
                                    source_repo=job_app.source or "",
                                )
                                async with semaphore:
                                    await process_discovered_job(listing, profile_content, client)
                                with lock:
                                    completed += 1
                                    progress.progress(completed / total)

                            async with _httpx.AsyncClient(timeout=30.0) as client:
                                tasks = [
                                    asyncio.create_task(_research_one(job_app))
                                    for job_app in jobs_missing_research
                                ]
                                await asyncio.gather(*tasks, return_exceptions=True)

                        asyncio.run(_run_research())
                    st.success(f"Phase 1 research complete for {total} jobs!")
                    st.rerun()

    st.markdown("---")

    # Queue preview
    st.subheader(f"📋 Queue Preview (Next {min(10, len(queued_jobs))} jobs)")
    for i, job in enumerate(queued_jobs[:10]):
        has_research = db.get_job_requirements(job.id) is not None
        research_badge = "🔬" if has_research else "⚠️"
        st.markdown(f"{i+1}. {research_badge} **{job.company}** - {job.role} ({job.location or 'Unknown'})")

    if len(queued_jobs) > 10:
        st.caption(f"... and {len(queued_jobs) - 10} more")

    st.markdown("---")

    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if not st.session_state.auto_apply_running:
            if st.button("▶️ Start Auto-Apply", use_container_width=True, type="primary"):
                st.session_state.auto_apply_running = True
                st.session_state.auto_apply_paused = False
                st.session_state.current_job_index = 0
                st.session_state.agent_logs = []
                add_log(f"Starting auto-apply for {len(queued_jobs)} jobs...", "action")
                st.rerun()
        else:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.auto_apply_running = False
                st.session_state.auto_apply_paused = False
                add_log("Auto-apply stopped by user", "warning")
                st.rerun()
    
    with col_btn2:
        if st.session_state.auto_apply_running:
            if st.session_state.auto_apply_paused:
                if st.button("▶️ Resume", use_container_width=True):
                    st.session_state.auto_apply_paused = False
                    add_log("Resuming auto-apply...", "action")
                    st.rerun()
            else:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.auto_apply_paused = True
                    add_log("Auto-apply paused - solve CAPTCHA if needed", "warning")
                    st.rerun()
    
    with col_btn3:
        if st.button("🗑️ Clear Queue", use_container_width=True):
            for job in queued_jobs:
                db.update_status(job.id, ApplicationStatus.SKIPPED)
            st.success("Queue cleared!")
            st.rerun()
    
    # Auto-apply execution
    if st.session_state.auto_apply_running and not st.session_state.auto_apply_paused:
        st.markdown("---")
        st.subheader("🤖 Agent Running...")
        
        # Progress bar
        progress = st.progress(0)
        status_text = st.empty()
        
        # Get fresh queue
        queued_jobs = db.get_all_applications(status=ApplicationStatus.QUEUED, limit=500)
        
        if not queued_jobs:
            st.session_state.auto_apply_running = False
            st.success("🎉 All jobs processed!")
            st.rerun()
        
        # Process next job
        current_job = queued_jobs[0]
        
        status_text.markdown(f"**Applying to:** {current_job.company} - {current_job.role}")
        add_log(f"Processing: {current_job.company} - {current_job.role}", "action")
        
        # Update status to in_progress
        db.update_status(current_job.id, ApplicationStatus.IN_PROGRESS)
        
        # Run the agent
        try:
            _sync_api_keys_to_env()
            # Check if we're resuming a session
            resume_session = st.session_state.get("pending_resume_session")
            if resume_session:
                st.session_state.pending_resume_session = None
                result = asyncio.run(run_agent_for_job(
                    job_url=current_job.job_url,
                    resume_path=resume_path,
                    company=current_job.company,
                    role=current_job.role,
                    job_id=current_job.id,
                    skip_duplicate=True,
                    resume_session_id=resume_session,
                ))
            else:
                result = asyncio.run(run_agent_for_job(
                    job_url=current_job.job_url,
                    resume_path=resume_path,
                    company=current_job.company,
                    role=current_job.role,
                    job_id=current_job.id,
                    skip_duplicate=True,
                ))
            
            if result.get("success"):
                db.update_status(current_job.id, ApplicationStatus.COMPLETED)
                add_log(f"✅ Completed: {current_job.company}", "success")
            else:
                # Check if it's a CAPTCHA situation
                error_msg = result.get("message", "").lower()
                if "captcha" in error_msg:
                    st.session_state.auto_apply_paused = True
                    # Store session ID for resume
                    if result.get("session_id"):
                        st.session_state.pending_resume_session = result.get("session_id")
                    add_log(f"⚠️ CAPTCHA detected at {current_job.company} - please solve manually", "warning")
                    # Desktop/sound notification is already sent by apply_agent
                    st.warning("🔐 **CAPTCHA Detected!** Please solve it in the browser, then click **Resume**.")
                elif result.get("can_resume"):
                    # Error but can resume - pause auto-apply to allow user to decide
                    st.session_state.auto_apply_paused = True
                    if result.get("session_id"):
                        st.session_state.pending_resume_session = result.get("session_id")
                    db.update_status(current_job.id, ApplicationStatus.IN_PROGRESS, notes=result.get("message"))
                    add_log(f"⚠️ Error at {current_job.company} - session saved for resume", "warning")
                    st.warning(f"⚠️ **Recoverable Error!** {result.get('message', 'Unknown error')}\n\nClick **Resume** to retry or **Stop** to skip this job.")
                else:
                    db.update_status(current_job.id, ApplicationStatus.FAILED, notes=result.get("message"))
                    add_log(f"❌ Failed: {current_job.company} - {result.get('message')}", "error")
        
        except Exception as e:
            db.update_status(current_job.id, ApplicationStatus.FAILED, notes=str(e))
            add_log(f"❌ Error: {current_job.company} - {str(e)}", "error")
        
        # Continue to next job if not paused
        if not st.session_state.auto_apply_paused:
            time.sleep(2)  # Brief pause between applications
            st.rerun()
    
    # Activity log
    st.markdown("---")
    st.subheader("📜 Activity Log")
    if st.session_state.agent_logs:
        log_text = "\n".join(reversed(st.session_state.agent_logs[-30:]))
        st.code(log_text, language=None)
    else:
        st.info("No activity yet. Start auto-apply to see logs.")


def render_manual_tab():
    """Render the manual application tab (Phase 2)."""
    st.header("📝 Manual Application")
    st.markdown("Apply to a single job by pasting its URL.")
    
    db = get_db()
    resume_path = get_resume_path()
    
    # Job URL Input
    job_url = st.text_input(
        "Job Posting URL",
        placeholder="https://boards.greenhouse.io/company/jobs/123456",
        help="Paste the direct link to the job application page"
    )
    
    # Check for duplicate and retrieve existing app ID (preserves Phase 1 data)
    is_duplicate = False
    existing_app = None
    if job_url:
        existing_app = db.get_application_by_url(job_url)
        if existing_app:
            is_duplicate = True
            st.warning(f"⚠️ Already applied to {existing_app.company} - {existing_app.role} on {existing_app.created_at.strftime('%Y-%m-%d')}")
    
    # Optional job details
    with st.expander("📝 Job Details (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company", placeholder="e.g., Google")
        with col2:
            role_title = st.text_input("Role", placeholder="e.g., Software Engineer")
    
    # Resume status
    if resume_path:
        st.success(f"✓ Using resume: {Path(resume_path).name}")
    else:
        st.error("❌ No resume found!")
    
    # Apply button
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_disabled = st.session_state.agent_running or not job_url or not resume_path
        button_label = "🔄 Re-apply Anyway" if is_duplicate else "🚀 Start Agent"
        
        if st.button(button_label, disabled=start_disabled, use_container_width=True):
            _sync_api_keys_to_env()
            st.session_state.agent_running = True
            st.session_state.agent_logs = []
            add_log(f"Starting agent for: {job_url}", "action")

            result = asyncio.run(run_agent_for_job(
                job_url=job_url,
                resume_path=resume_path,
                company=company_name or (existing_app.company if existing_app else None),
                role=role_title or (existing_app.role if existing_app else None),
                job_id=existing_app.id if existing_app else None,
                skip_duplicate=is_duplicate,
            ))
            
            st.session_state.agent_running = False
            
            if result.get("success"):
                st.success("✅ Agent completed! Review the application in the browser.")
            elif result.get("can_resume") and result.get("session_id"):
                st.session_state.pending_resume_session = result.get("session_id")
                st.warning(f"⚠️ Agent stopped: {result.get('message')}\n\nSession saved - click **Resume Session** to continue.")
            else:
                st.error(f"❌ Agent stopped: {result.get('message')}")
            
            st.rerun()
    
    with col2:
        # Resume button if there's a pending session
        resume_disabled = not st.session_state.pending_resume_session or st.session_state.agent_running
        if st.button("🔄 Resume Session", disabled=resume_disabled, use_container_width=True):
            _sync_api_keys_to_env()
            st.session_state.agent_running = True
            add_log("Resuming session...", "action")

            result = asyncio.run(run_agent_for_job(
                job_url=job_url or "",
                resume_path=resume_path or "",
                job_id=existing_app.id if existing_app else None,
                resume_session_id=st.session_state.pending_resume_session,
            ))
            
            st.session_state.agent_running = False
            
            if result.get("success"):
                st.session_state.pending_resume_session = None
                st.success("✅ Agent completed! Review the application in the browser.")
            elif result.get("can_resume") and result.get("session_id"):
                st.session_state.pending_resume_session = result.get("session_id")
                st.warning(f"⚠️ Agent stopped: {result.get('message')}")
            else:
                st.session_state.pending_resume_session = None
                st.error(f"❌ Agent stopped: {result.get('message')}")
            
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.agent_logs = []
            st.session_state.pending_resume_session = None
            st.rerun()
    
    # Activity log
    st.markdown("---")
    st.subheader("📜 Activity Log")
    if st.session_state.agent_logs:
        log_text = "\n".join(reversed(st.session_state.agent_logs[-30:]))
        st.code(log_text, language=None)


def render_history_tab():
    """Render the application history tab."""
    st.header("📚 Application History")
    
    db = get_db()
    
    # Filter options
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        status_filter = st.selectbox(
            "Status",
            options=["All", "queued", "in_progress", "completed", "submitted", "failed", "interview", "offer", "rejected", "skipped"]
        )
    
    with col2:
        search_query = st.text_input("🔍 Search", placeholder="Company or role...")
    
    with col3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Get applications
    if search_query:
        applications = db.search_applications(search_query)
    elif status_filter != "All":
        applications = db.get_all_applications(status=ApplicationStatus(status_filter), limit=200)
    else:
        applications = db.get_all_applications(limit=200)
    
    # Stats row
    st.markdown("---")
    stats = db.get_statistics()
    cols = st.columns(6)
    status_counts = stats.get("by_status", {})
    
    status_display = [
        ("queued", "⏳"), ("completed", "✅"), ("submitted", "📤"),
        ("interview", "🎯"), ("offer", "🎉"), ("rejected", "👎")
    ]
    
    for i, (status, emoji) in enumerate(status_display):
        with cols[i]:
            count = status_counts.get(status, 0)
            st.metric(f"{emoji} {status.title()}", count)
    
    # Application list
    st.markdown("---")
    
    if applications:
        for app in applications:
            status_emoji = {
                "queued": "⏳", "in_progress": "🔄", "completed": "✅",
                "submitted": "📤", "failed": "❌", "rejected": "👎",
                "interview": "🎯", "offer": "🎉", "skipped": "⏭️"
            }.get(app.status.value, "❓")
            
            with st.expander(f"{status_emoji} {app.company} - {app.role}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**URL:** [{app.job_url[:50]}...]({app.job_url})")
                    st.markdown(f"**Location:** {app.location or 'Unknown'}")
                    st.markdown(f"**Created:** {app.created_at.strftime('%Y-%m-%d %H:%M')}")
                    if app.applied_at:
                        st.markdown(f"**Applied:** {app.applied_at.strftime('%Y-%m-%d %H:%M')}")
                    if app.notes:
                        st.markdown(f"**Notes:** {app.notes}")
                
                with col2:
                    st.markdown("**Update Status:**")
                    new_status = st.selectbox(
                        "Status",
                        options=["submitted", "interview", "offer", "rejected", "failed"],
                        key=f"status_{app.id}",
                        label_visibility="collapsed"
                    )
                    if st.button("Update", key=f"update_{app.id}"):
                        db.update_status(app.id, ApplicationStatus(new_status))
                        st.rerun()
                    
                    if st.button("🗑️ Delete", key=f"delete_{app.id}"):
                        db.delete_application(app.id)
                        st.rerun()
    else:
        st.info("No applications found. Start applying to build your history!")


def main():
    """Main application entry point."""
    init_session_state()
    
    # Initialize notification config
    update_notification_config()
    
    # Start daily summary scheduler
    start_notification_scheduler()
    
    # Header
    st.markdown('<p class="main-header">🤖 AI Job Application Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Two-phase cloud pipeline • Groq 70B research + Qwen 32B browser automation</p>', unsafe_allow_html=True)
    
    # Check for pending session resume at startup
    if st.session_state.pending_resume_session:
        session_id = st.session_state.pending_resume_session
        state_manager = get_state_manager()
        state = state_manager.load_state(session_id)
        if state:
            st.info(f"🔄 Ready to resume: **{state.role}** at **{state.company}**")
    
    # Sidebar (returns nothing — model is now fixed per phase)
    render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Discover", "🚀 Auto-Apply", "📝 Manual", "📚 History"])

    with tab1:
        render_discover_tab()

    with tab2:
        render_auto_apply_tab()

    with tab3:
        render_manual_tab()

    with tab4:
        render_history_tab()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.875rem;">
        🔒 Personal data stays local • Phase 1: Groq Llama-3.3-70B • Phase 2: Qwen2.5-Coder-32B on cloud GPU
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
