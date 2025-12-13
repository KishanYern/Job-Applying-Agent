"""
Local AI Job Application Agent - Streamlit UI
A privacy-focused, automated job application tool running entirely on local hardware.
"""

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


async def run_agent_for_job(
    job_url: str,
    resume_path: str,
    model_name: str,
    company: str = None,
    role: str = None,
    skip_duplicate: bool = False,
    resume_session_id: str = None
) -> dict:
    """Run the job application agent for a single job."""
    try:
        from apply_agent import run_agent, resume_agent
        
        # If resuming a session, use the resume function
        if resume_session_id:
            result = await resume_agent(
                session_id=resume_session_id,
                log_callback=add_log
            )
        else:
            result = await run_agent(
                job_url=job_url,
                resume_path=resume_path,
                profile_path=str(Path(__file__).parent / "my_profile.md"),
                model_name=model_name,
                log_callback=add_log,
                company=company,
                role=role,
                skip_duplicate_check=skip_duplicate
            )
        return result
    except Exception as e:
        return {"success": False, "message": str(e), "can_resume": False}


def render_sidebar(model_name_key: str = "model_select"):
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        model_name = st.selectbox(
            "Select LLM Model",
            options=["qwen2.5:7b", "llama3.1", "llama3.1:8b", "llama3.2", "llama3.2:3b"],
            index=0,
            help="Choose the Ollama model for the agent's reasoning",
            key=model_name_key
        )
        
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
        - **🔍 Discover**: Find jobs from GitHub
        - **🚀 Auto-Apply**: Batch apply to queue
        - **📝 Manual**: Apply to single URL
        - **📚 History**: Track applications
        """)
    
    return model_name


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
    
    # Search button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        search_clicked = st.button("🔍 Search Jobs", use_container_width=True)
    
    if search_clicked:
        with st.spinner("Fetching jobs from GitHub repositories..."):
            locations = [l.strip() for l in location_filter.split(",")] if location_filter else None
            keywords = [k.strip() for k in keyword_filter.split(",")] if keyword_filter else None
            
            jobs = get_job_urls_sync(
                keywords=keywords,
                locations=locations,
                job_type=job_type if job_type != "all" else None
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


def render_auto_apply_tab(model_name: str):
    """Render the auto-apply tab."""
    st.header("🚀 Auto-Apply Mode")
    st.markdown("Automatically apply to all queued jobs. You only intervene for CAPTCHAs.")
    
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
    
    # Queue preview
    st.subheader(f"📋 Queue Preview (Next {min(10, len(queued_jobs))} jobs)")
    for i, job in enumerate(queued_jobs[:10]):
        st.markdown(f"{i+1}. **{job.company}** - {job.role} ({job.location or 'Unknown'})")
    
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
            # Check if we're resuming a session
            resume_session = st.session_state.get("pending_resume_session")
            if resume_session:
                st.session_state.pending_resume_session = None
                result = asyncio.run(run_agent_for_job(
                    job_url=current_job.job_url,
                    resume_path=resume_path,
                    model_name=model_name,
                    company=current_job.company,
                    role=current_job.role,
                    skip_duplicate=True,
                    resume_session_id=resume_session
                ))
            else:
                result = asyncio.run(run_agent_for_job(
                    job_url=current_job.job_url,
                    resume_path=resume_path,
                    model_name=model_name,
                    company=current_job.company,
                    role=current_job.role,
                    skip_duplicate=True
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


def render_manual_tab(model_name: str):
    """Render the manual application tab."""
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
    
    # Check for duplicate
    is_duplicate = False
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
            st.session_state.agent_running = True
            st.session_state.agent_logs = []
            add_log(f"Starting agent for: {job_url}", "action")
            
            result = asyncio.run(run_agent_for_job(
                job_url=job_url,
                resume_path=resume_path,
                model_name=model_name,
                company=company_name or None,
                role=role_title or None,
                skip_duplicate=is_duplicate
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
            st.session_state.agent_running = True
            add_log(f"Resuming session...", "action")
            
            result = asyncio.run(run_agent_for_job(
                job_url=job_url or "",
                resume_path=resume_path,
                model_name=model_name,
                resume_session_id=st.session_state.pending_resume_session
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
    st.markdown('<p class="sub-header">Fully automated job applications • Privacy-first • 100% local</p>', unsafe_allow_html=True)
    
    # Check for pending session resume at startup
    if st.session_state.pending_resume_session:
        session_id = st.session_state.pending_resume_session
        state_manager = get_state_manager()
        state = state_manager.load_state(session_id)
        if state:
            st.info(f"🔄 Ready to resume: **{state.role}** at **{state.company}**")
    
    # Sidebar
    model_name = render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Discover", "🚀 Auto-Apply", "📝 Manual", "📚 History"])
    
    with tab1:
        render_discover_tab()
    
    with tab2:
        render_auto_apply_tab(model_name)
    
    with tab3:
        render_manual_tab(model_name)
    
    with tab4:
        render_history_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.875rem;">
        🔒 All data stays on your machine • Powered by Ollama • No external APIs
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
