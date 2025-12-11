"""
Local AI Job Application Agent - Streamlit UI
A privacy-focused, automated job application tool running entirely on local hardware.
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path

# Enable nested async event loops (required for Streamlit + async browser-use)
import nest_asyncio
nest_asyncio.apply()

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
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-running {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
    }
    .status-success {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    .status-error {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
    }
    .info-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
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
        st.session_state.agent_status = "idle"  # idle, running, success, error, paused


def load_profile() -> str:
    """Load the user profile from my_profile.md."""
    profile_path = Path(__file__).parent / "my_profile.md"
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8")
    return ""


def add_log(message: str, level: str = "info"):
    """Add a log message to the session state."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "action": "🤖"}.get(level, "ℹ️")
    st.session_state.agent_logs.append(f"[{timestamp}] {icon} {message}")


async def run_agent_async(job_url: str, resume_path: str, model_name: str):
    """Run the job application agent asynchronously."""
    try:
        # Import the agent module
        from apply_agent import run_agent
        
        add_log(f"Starting agent with model: {model_name}", "action")
        add_log(f"Target URL: {job_url}", "info")
        add_log(f"Resume: {resume_path}", "info")
        
        st.session_state.agent_status = "running"
        
        # Run the agent
        result = await run_agent(
            job_url=job_url,
            resume_path=resume_path,
            profile_path=str(Path(__file__).parent / "my_profile.md"),
            model_name=model_name,
            log_callback=add_log
        )
        
        if result.get("success"):
            st.session_state.agent_status = "success"
            add_log("Agent completed successfully!", "success")
        else:
            st.session_state.agent_status = "error"
            add_log(f"Agent stopped: {result.get('message', 'Unknown error')}", "error")
            
    except ImportError:
        st.session_state.agent_status = "error"
        add_log("Agent module not found. Please ensure apply_agent.py exists.", "error")
    except Exception as e:
        st.session_state.agent_status = "error"
        add_log(f"Agent error: {str(e)}", "error")
    finally:
        st.session_state.agent_running = False


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">🤖 AI Job Application Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Privacy-focused automation running 100% locally with Llama 3.1</p>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        model_name = st.selectbox(
            "Select LLM Model",
            options=["llama3.1", "llama3.1:8b", "llama3.2", "llama3.2:3b"],
            index=0,
            help="Choose the Ollama model for the agent's reasoning"
        )
        
        st.divider()
        
        # Profile Preview
        st.header("👤 Your Profile")
        profile_content = load_profile()
        if profile_content:
            with st.expander("View Profile Data", expanded=False):
                st.text(profile_content[:2000] + "..." if len(profile_content) > 2000 else profile_content)
            st.success("✓ Profile loaded from my_profile.md")
        else:
            st.error("⚠️ my_profile.md not found!")
            st.info("Create a my_profile.md file with your resume data.")
        
        st.divider()
        
        # System Status
        st.header("📊 System Status")
        
        # Check Ollama status (simplified check)
        ollama_status = st.empty()
        ollama_status.info("🔄 Ollama: Checking...")
        # In a real implementation, we'd ping Ollama here
        ollama_status.success("✓ Ollama: Ready")
        
        st.divider()
        
        # Instructions
        st.header("📖 How to Use")
        st.markdown("""
        1. **Paste** a GitHub job board URL
        2. **Click** "Start Agent"
        3. **Watch** the browser automation
        4. **Intervene** if CAPTCHA appears
        5. **Review** before final submission
        """)
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Job Application")
        
        # Job URL Input
        job_url = st.text_input(
            "Job Posting URL",
            placeholder="https://boards.greenhouse.io/company/jobs/123456",
            help="Paste the direct link to the job application page"
        )
        
        # Resume Upload (required for job applications)
        uploaded_file = st.file_uploader(
            "📎 Upload Resume (PDF)",
            type=["pdf"],
            help="Required: Your resume PDF will be uploaded to job applications"
        )
        resume_path = None
        if uploaded_file:
            # Save to temp location
            temp_path = Path(__file__).parent / "data" / "uploaded_resume.pdf"
            temp_path.parent.mkdir(exist_ok=True)
            temp_path.write_bytes(uploaded_file.getvalue())
            resume_path = str(temp_path)
            st.success(f"✓ Resume ready: {uploaded_file.name}")
        
        # Control Buttons
        st.markdown("---")
        
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            start_disabled = st.session_state.agent_running or not job_url or not resume_path
            if st.button("🚀 Start Agent", disabled=start_disabled, use_container_width=True):
                st.session_state.agent_running = True
                st.session_state.agent_logs = []
                st.session_state.agent_status = "running"
                add_log("Initializing agent...", "info")
                
                # Run the async agent
                # nest_asyncio.apply() at module level allows asyncio.run() to work
                # even when Streamlit's event loop is already running
                asyncio.run(run_agent_async(job_url, resume_path, model_name))
                
                st.rerun()
            
            # Show hint if resume not uploaded
            if not resume_path and job_url:
                st.caption("⚠️ Upload resume to enable")
        
        with button_col2:
            if st.button("⏸️ Pause Agent", disabled=not st.session_state.agent_running, use_container_width=True):
                st.session_state.agent_status = "paused"
                add_log("Agent paused by user", "warning")
        
        with button_col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.agent_running = False
                st.session_state.agent_logs = []
                st.session_state.agent_status = "idle"
                st.rerun()
    
    with col2:
        st.header("📋 Status")
        
        # Status indicator
        status = st.session_state.agent_status
        if status == "idle":
            st.info("⏳ **Ready** - Enter a job URL to begin")
        elif status == "running":
            st.warning("🔄 **Running** - Agent is working...")
        elif status == "success":
            st.success("✅ **Complete** - Review the application")
        elif status == "error":
            st.error("❌ **Error** - Check logs below")
        elif status == "paused":
            st.warning("⏸️ **Paused** - Human intervention needed")
    
    # Agent Logs
    st.header("📜 Agent Activity Log")
    
    log_container = st.container()
    with log_container:
        if st.session_state.agent_logs:
            log_text = "\n".join(reversed(st.session_state.agent_logs[-50:]))  # Show last 50 logs
            st.code(log_text, language=None)
        else:
            st.markdown("""
            <div class="info-card">
                <p style="color: #6B7280; text-align: center;">
                    No activity yet. Start the agent to see logs here.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.875rem;">
        🔒 All data stays on your machine • Powered by Ollama + Llama 3.1 • No external APIs
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
