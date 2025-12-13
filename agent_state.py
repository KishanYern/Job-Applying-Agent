"""
Agent State Management - Save and recover application state for error recovery.
Handles browser crashes, network failures, and interrupted sessions.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum


class AgentPhase(Enum):
    """Phases of the application process."""
    INITIALIZING = "initializing"
    NAVIGATING = "navigating"
    ANALYZING_PAGE = "analyzing_page"
    FILLING_FORM = "filling_form"
    UPLOADING_RESUME = "uploading_resume"
    RESEARCHING = "researching"  # Multi-tab research
    REVIEWING = "reviewing"
    PAUSED_CAPTCHA = "paused_captcha"
    PAUSED_USER = "paused_user"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FormFieldState:
    """State of a single form field."""
    field_name: str
    field_type: str  # text, select, checkbox, file, etc.
    value: Optional[str] = None
    filled: bool = False
    selector: Optional[str] = None  # CSS selector if available


@dataclass
class AgentState:
    """
    Complete state of an agent run.
    Can be serialized and restored to resume after crashes.
    """
    # Identity
    session_id: str
    job_url: str
    application_id: Optional[int] = None
    
    # Job metadata
    company: str = "Unknown"
    role: str = "Unknown"
    
    # Progress tracking
    phase: AgentPhase = AgentPhase.INITIALIZING
    current_page_url: Optional[str] = None
    page_number: int = 1
    total_pages: Optional[int] = None
    
    # Form state
    fields_filled: Dict[str, FormFieldState] = field(default_factory=dict)
    fields_total: int = 0
    resume_uploaded: bool = False
    
    # Research state (for multi-tab research)
    research_queries: List[str] = field(default_factory=list)
    research_results: Dict[str, str] = field(default_factory=dict)
    
    # Error tracking
    retry_count: int = 0
    last_error: Optional[str] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    started_at: str = ""
    updated_at: str = ""
    last_action: str = ""
    
    # Configuration used
    model_name: str = ""
    resume_path: str = ""
    profile_path: str = ""
    
    def __post_init__(self):
        """Initialize timestamps if not set."""
        now = datetime.now().isoformat()
        if not self.started_at:
            self.started_at = now
        self.updated_at = now
    
    def update_phase(self, phase: AgentPhase, action: str = "") -> None:
        """Update the current phase and timestamp."""
        self.phase = phase
        self.updated_at = datetime.now().isoformat()
        if action:
            self.last_action = action
    
    def record_field_filled(
        self, 
        field_name: str, 
        field_type: str, 
        value: str,
        selector: Optional[str] = None
    ) -> None:
        """Record that a form field was filled."""
        self.fields_filled[field_name] = FormFieldState(
            field_name=field_name,
            field_type=field_type,
            value=value,
            filled=True,
            selector=selector
        )
        self.updated_at = datetime.now().isoformat()
        self.last_action = f"Filled: {field_name}"
    
    def record_error(self, error_msg: str, error_type: str = "unknown") -> None:
        """Record an error in the history."""
        self.last_error = error_msg
        self.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_msg,
            "phase": self.phase.value,
            "retry_count": self.retry_count
        })
        self.updated_at = datetime.now().isoformat()
    
    def increment_retry(self) -> int:
        """Increment retry count and return new value."""
        self.retry_count += 1
        self.updated_at = datetime.now().isoformat()
        return self.retry_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["phase"] = self.phase.value
        data["fields_filled"] = {
            k: asdict(v) for k, v in self.fields_filled.items()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create from dictionary."""
        # Convert phase back to enum
        if "phase" in data and isinstance(data["phase"], str):
            data["phase"] = AgentPhase(data["phase"])
        
        # Convert fields_filled back to FormFieldState objects
        if "fields_filled" in data:
            data["fields_filled"] = {
                k: FormFieldState(**v) for k, v in data["fields_filled"].items()
            }
        
        return cls(**data)
    
    def get_progress_percent(self) -> float:
        """Get completion percentage."""
        if self.fields_total == 0:
            return 0.0
        filled = len([f for f in self.fields_filled.values() if f.filled])
        return (filled / self.fields_total) * 100


class StateManager:
    """
    Manages agent state persistence and recovery.
    Saves state to disk periodically and on key events.
    """
    
    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize the state manager.
        
        Args:
            state_dir: Directory to store state files.
                      Defaults to 'data/agent_state/' in project directory.
        """
        if state_dir is None:
            state_dir = str(Path(__file__).parent / "data" / "agent_state")
        
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_state: Optional[AgentState] = None
        self._auto_save_interval = 5  # seconds
        self._last_save_time = 0.0
    
    def _get_state_file(self, session_id: str) -> Path:
        """Get the path to a state file."""
        return self.state_dir / f"{session_id}.json"
    
    def _get_active_state_file(self) -> Path:
        """Get the path to the 'active' state marker file."""
        return self.state_dir / "active_session.txt"
    
    def create_session(
        self,
        job_url: str,
        company: str = "Unknown",
        role: str = "Unknown",
        application_id: Optional[int] = None,
        model_name: str = "",
        resume_path: str = "",
        profile_path: str = ""
    ) -> AgentState:
        """
        Create a new agent session state.
        
        Returns:
            New AgentState instance
        """
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}_{hash(job_url) % 10000:04d}"
        
        state = AgentState(
            session_id=session_id,
            job_url=job_url,
            company=company,
            role=role,
            application_id=application_id,
            model_name=model_name,
            resume_path=resume_path,
            profile_path=profile_path
        )
        
        self._current_state = state
        self.save_state(state)
        
        # Mark as active session
        self._get_active_state_file().write_text(session_id)
        
        return state
    
    def save_state(self, state: Optional[AgentState] = None) -> bool:
        """
        Save the agent state to disk.
        
        Args:
            state: State to save. Uses current state if not provided.
        
        Returns:
            True if saved successfully
        """
        state = state or self._current_state
        if state is None:
            return False
        
        try:
            state_file = self._get_state_file(state.session_id)
            state.updated_at = datetime.now().isoformat()
            
            with open(state_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            
            self._last_save_time = time.time()
            return True
            
        except Exception as e:
            print(f"Failed to save state: {e}")
            return False
    
    def auto_save(self, state: Optional[AgentState] = None) -> bool:
        """
        Auto-save if enough time has passed since last save.
        Call this frequently during agent execution.
        """
        if time.time() - self._last_save_time >= self._auto_save_interval:
            return self.save_state(state)
        return False
    
    def load_state(self, session_id: str) -> Optional[AgentState]:
        """
        Load a saved state by session ID.
        
        Returns:
            AgentState if found, None otherwise
        """
        try:
            state_file = self._get_state_file(session_id)
            if not state_file.exists():
                return None
            
            with open(state_file, "r") as f:
                data = json.load(f)
            
            state = AgentState.from_dict(data)
            self._current_state = state
            return state
            
        except Exception as e:
            print(f"Failed to load state: {e}")
            return None
    
    def get_active_session(self) -> Optional[AgentState]:
        """
        Get the currently active session (if any).
        Used for crash recovery.
        
        Returns:
            AgentState if there's an active session, None otherwise
        """
        try:
            active_file = self._get_active_state_file()
            if not active_file.exists():
                return None
            
            session_id = active_file.read_text().strip()
            if not session_id:
                return None
            
            state = self.load_state(session_id)
            
            # Check if the session was interrupted (not completed or failed)
            if state and state.phase not in [AgentPhase.COMPLETED, AgentPhase.FAILED]:
                return state
            
            return None
            
        except Exception:
            return None
    
    def get_recoverable_sessions(self, max_age_hours: int = 24) -> List[AgentState]:
        """
        Get all sessions that can be recovered.
        
        Args:
            max_age_hours: Maximum age of sessions to consider
        
        Returns:
            List of recoverable AgentState objects
        """
        recoverable = []
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        
        try:
            for state_file in self.state_dir.glob("session_*.json"):
                # Check file age
                if state_file.stat().st_mtime < cutoff:
                    continue
                
                try:
                    with open(state_file, "r") as f:
                        data = json.load(f)
                    
                    state = AgentState.from_dict(data)
                    
                    # Only include interrupted sessions
                    if state.phase not in [AgentPhase.COMPLETED, AgentPhase.FAILED]:
                        recoverable.append(state)
                        
                except Exception:
                    continue
            
            # Sort by most recent first
            recoverable.sort(key=lambda s: s.updated_at, reverse=True)
            return recoverable
            
        except Exception:
            return []
    
    def mark_completed(self, state: Optional[AgentState] = None) -> None:
        """Mark a session as completed."""
        state = state or self._current_state
        if state:
            state.update_phase(AgentPhase.COMPLETED, "Application completed")
            self.save_state(state)
        
        # Clear active session marker
        self.clear_active_session()
    
    def mark_failed(self, state: Optional[AgentState] = None, error: str = "") -> None:
        """Mark a session as failed."""
        state = state or self._current_state
        if state:
            state.update_phase(AgentPhase.FAILED, f"Failed: {error}")
            if error:
                state.record_error(error, "fatal")
            self.save_state(state)
        
        # Clear active session marker
        self.clear_active_session()
    
    def clear_active_session(self) -> None:
        """Clear the active session marker."""
        try:
            active_file = self._get_active_state_file()
            if active_file.exists():
                active_file.unlink()
        except Exception:
            pass
    
    def cleanup_old_sessions(self, max_age_days: int = 7) -> int:
        """
        Remove old session files.
        
        Args:
            max_age_days: Delete sessions older than this
        
        Returns:
            Number of sessions deleted
        """
        deleted = 0
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)
        
        try:
            for state_file in self.state_dir.glob("session_*.json"):
                if state_file.stat().st_mtime < cutoff:
                    state_file.unlink()
                    deleted += 1
        except Exception:
            pass
        
        return deleted
    
    @property
    def current_state(self) -> Optional[AgentState]:
        """Get the current state."""
        return self._current_state
    
    @current_state.setter
    def current_state(self, state: AgentState) -> None:
        """Set the current state."""
        self._current_state = state


# Singleton instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the singleton state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


# Retry configuration and utilities
@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: float = 0.1  # Random jitter factor
    
    # Specific retry settings
    network_retries: int = 5
    browser_restart_retries: int = 2


def calculate_retry_delay(
    retry_count: int,
    config: RetryConfig = RetryConfig()
) -> float:
    """
    Calculate delay before next retry using exponential backoff.
    
    Args:
        retry_count: Current retry attempt (0-indexed)
        config: Retry configuration
    
    Returns:
        Delay in seconds
    """
    import random
    
    # Exponential backoff
    delay = config.initial_delay * (config.exponential_base ** retry_count)
    
    # Cap at max delay
    delay = min(delay, config.max_delay)
    
    # Add jitter
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


def is_retriable_error(error: Exception) -> bool:
    """
    Determine if an error is retriable.
    
    Args:
        error: The exception that occurred
    
    Returns:
        True if the error can be retried
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Network-related errors (retriable)
    retriable_patterns = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporary",
        "unavailable",
        "reset by peer",
        "broken pipe",
        "eof",
        "socket",
        "refused",
        "dns",
        "resolve",
    ]
    
    for pattern in retriable_patterns:
        if pattern in error_str or pattern in error_type:
            return True
    
    # Browser-specific retriable errors
    browser_retriable = [
        "target closed",
        "browser",
        "page crashed",
        "context",
        "frame was detached",
    ]
    
    for pattern in browser_retriable:
        if pattern in error_str:
            return True
    
    return False


def is_browser_crash(error: Exception) -> bool:
    """
    Determine if an error indicates a browser crash.
    
    Args:
        error: The exception that occurred
    
    Returns:
        True if the error indicates a browser crash
    """
    error_str = str(error).lower()
    
    crash_patterns = [
        "browser",
        "target closed",
        "page crashed",
        "context closed",
        "connection closed",
        "playwright",
    ]
    
    for pattern in crash_patterns:
        if pattern in error_str:
            return True
    
    return False


def is_captcha_error(error: Exception) -> bool:
    """Check if error is related to CAPTCHA detection."""
    error_str = str(error).lower()
    return "captcha" in error_str or "recaptcha" in error_str or "hcaptcha" in error_str


# CLI for testing
if __name__ == "__main__":
    import sys
    
    manager = get_state_manager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            sessions = manager.get_recoverable_sessions()
            if sessions:
                print(f"\n📋 Recoverable Sessions ({len(sessions)}):\n")
                for s in sessions:
                    print(f"  • {s.session_id}")
                    print(f"    Job: {s.role} at {s.company}")
                    print(f"    Phase: {s.phase.value}")
                    print(f"    Progress: {s.get_progress_percent():.0f}%")
                    print(f"    Updated: {s.updated_at}")
                    print()
            else:
                print("No recoverable sessions found.")
        
        elif command == "active":
            state = manager.get_active_session()
            if state:
                print(f"\n🔄 Active Session Found:\n")
                print(f"  Session: {state.session_id}")
                print(f"  Job: {state.role} at {state.company}")
                print(f"  URL: {state.job_url}")
                print(f"  Phase: {state.phase.value}")
                print(f"  Retries: {state.retry_count}")
                if state.last_error:
                    print(f"  Last Error: {state.last_error}")
            else:
                print("No active session.")
        
        elif command == "cleanup":
            deleted = manager.cleanup_old_sessions()
            print(f"🗑️ Cleaned up {deleted} old sessions")
        
        elif command == "test":
            # Create a test session
            state = manager.create_session(
                job_url="https://example.com/apply/12345",
                company="Test Corp",
                role="Software Engineer"
            )
            print(f"✅ Created test session: {state.session_id}")
            
            # Simulate progress
            state.update_phase(AgentPhase.FILLING_FORM, "Started form")
            state.record_field_filled("first_name", "text", "John")
            state.record_field_filled("last_name", "text", "Doe")
            state.fields_total = 10
            manager.save_state()
            
            print(f"   Progress: {state.get_progress_percent():.0f}%")
            print(f"   Saved to: {manager._get_state_file(state.session_id)}")
        
        else:
            print(f"Unknown command: {command}")
    
    else:
        print("\nAgent State Manager")
        print("==================")
        print("\nUsage:")
        print("  python agent_state.py list     - List recoverable sessions")
        print("  python agent_state.py active   - Show active session")
        print("  python agent_state.py cleanup  - Remove old sessions")
        print("  python agent_state.py test     - Create a test session")
