"""
Notification System - Desktop and sound alerts for the Job Application Agent.
Provides cross-platform notifications for CAPTCHAs, errors, and daily summaries.
"""

import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum


class NotificationType(Enum):
    """Types of notifications."""
    CAPTCHA = "captcha"
    ERROR = "error"
    SUCCESS = "success"
    DAILY_SUMMARY = "daily_summary"
    BROWSER_CRASH = "browser_crash"
    NETWORK_ERROR = "network_error"
    INFO = "info"


@dataclass
class NotificationConfig:
    """Configuration for notification behavior."""
    enable_desktop: bool = True
    enable_sound: bool = True
    sound_volume: float = 0.8  # 0.0 to 1.0
    daily_summary_time: str = "18:00"  # 24-hour format
    
    # Sound files (relative to project root or absolute)
    captcha_sound: str = "sounds/captcha_alert.wav"
    error_sound: str = "sounds/error.wav"
    success_sound: str = "sounds/success.wav"


# Default configuration
_config = NotificationConfig()


def configure_notifications(config: NotificationConfig) -> None:
    """Update the notification configuration."""
    global _config
    _config = config


def get_config() -> NotificationConfig:
    """Get current notification configuration."""
    return _config


def _get_sound_path(sound_file: str) -> Optional[str]:
    """Get the absolute path to a sound file."""
    # Check if it's already absolute
    if os.path.isabs(sound_file):
        return sound_file if os.path.exists(sound_file) else None
    
    # Check relative to project root
    project_root = Path(__file__).parent
    sound_path = project_root / sound_file
    
    if sound_path.exists():
        return str(sound_path)
    
    return None


def play_sound(notification_type: NotificationType) -> bool:
    """
    Play an alert sound for the given notification type.
    Uses system sounds as fallback if custom sounds don't exist.
    
    Returns True if sound was played successfully.
    """
    if not _config.enable_sound:
        return False
    
    # Map notification type to sound file
    sound_map = {
        NotificationType.CAPTCHA: _config.captcha_sound,
        NotificationType.ERROR: _config.error_sound,
        NotificationType.BROWSER_CRASH: _config.error_sound,
        NotificationType.NETWORK_ERROR: _config.error_sound,
        NotificationType.SUCCESS: _config.success_sound,
        NotificationType.DAILY_SUMMARY: _config.success_sound,
    }
    
    sound_file = sound_map.get(notification_type)
    sound_path = _get_sound_path(sound_file) if sound_file else None
    
    # Try to play custom sound first
    if sound_path:
        try:
            from playsound import playsound
            # Run in thread to avoid blocking
            thread = threading.Thread(target=playsound, args=(sound_path,), daemon=True)
            thread.start()
            return True
        except Exception:
            pass
    
    # Fallback: use system beep
    return _play_system_beep(notification_type)


def _play_system_beep(notification_type: NotificationType) -> bool:
    """Play system beep as fallback."""
    try:
        if sys.platform == "linux":
            # Try different methods for Linux
            beep_count = 3 if notification_type == NotificationType.CAPTCHA else 1
            
            # Method 1: paplay (PulseAudio)
            for _ in range(beep_count):
                os.system('paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null || '
                         'paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null &')
            return True
            
        elif sys.platform == "darwin":
            # macOS: use afplay or say
            beep_count = 3 if notification_type == NotificationType.CAPTCHA else 1
            for _ in range(beep_count):
                os.system('afplay /System/Library/Sounds/Ping.aiff &')
            return True
            
        elif sys.platform == "win32":
            # Windows: use winsound
            import winsound
            freq = 1000 if notification_type == NotificationType.CAPTCHA else 800
            duration = 500
            beep_count = 3 if notification_type == NotificationType.CAPTCHA else 1
            for _ in range(beep_count):
                winsound.Beep(freq, duration)
            return True
            
    except Exception:
        pass
    
    return False


def send_desktop_notification(
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.INFO,
    timeout: int = 10
) -> bool:
    """
    Send a desktop notification.
    
    Args:
        title: Notification title
        message: Notification body
        notification_type: Type of notification (affects icon)
        timeout: How long to show notification (seconds)
    
    Returns True if notification was sent successfully.
    """
    if not _config.enable_desktop:
        return False
    
    # Try plyer first (cross-platform)
    try:
        from plyer import notification as plyer_notification
        
        # Map type to icon
        icon_map = {
            NotificationType.CAPTCHA: "dialog-warning",
            NotificationType.ERROR: "dialog-error",
            NotificationType.BROWSER_CRASH: "dialog-error",
            NotificationType.NETWORK_ERROR: "dialog-warning",
            NotificationType.SUCCESS: "dialog-information",
            NotificationType.DAILY_SUMMARY: "dialog-information",
            NotificationType.INFO: "dialog-information",
        }
        
        plyer_notification.notify(
            title=title,
            message=message,
            app_name="Job Application Agent",
            timeout=timeout
        )
        return True
        
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: platform-specific methods
    return _send_platform_notification(title, message, notification_type)


def _send_platform_notification(
    title: str,
    message: str,
    notification_type: NotificationType
) -> bool:
    """Send notification using platform-specific methods."""
    try:
        if sys.platform == "linux":
            # Use notify-send
            import subprocess
            urgency = "critical" if notification_type == NotificationType.CAPTCHA else "normal"
            subprocess.run([
                "notify-send",
                "--urgency", urgency,
                "--app-name", "Job Application Agent",
                title,
                message
            ], check=False)
            return True
            
        elif sys.platform == "darwin":
            # macOS: use osascript
            import subprocess
            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], check=False)
            return True
            
        elif sys.platform == "win32":
            # Windows: use win10toast if available, otherwise basic messagebox
            try:
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(title, message, duration=5, threaded=True)
                return True
            except ImportError:
                pass
                
    except Exception:
        pass
    
    return False


def notify(
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.INFO,
    play_alert: bool = True
) -> None:
    """
    Send a notification with both desktop alert and sound.
    
    Args:
        title: Notification title
        message: Notification body
        notification_type: Type of notification
        play_alert: Whether to also play a sound
    """
    # Send desktop notification
    send_desktop_notification(title, message, notification_type)
    
    # Play sound if requested
    if play_alert:
        play_sound(notification_type)


def notify_captcha(company: str = "Unknown", url: str = "") -> None:
    """Send CAPTCHA alert notification."""
    title = "🔐 CAPTCHA Detected!"
    message = f"Manual intervention needed at {company}"
    if url:
        message += f"\n{url[:50]}..."
    
    notify(title, message, NotificationType.CAPTCHA, play_alert=True)
    
    # Log to console as well
    print(f"\n{'='*60}")
    print(f"🔐 CAPTCHA ALERT: {company}")
    print(f"Please solve the CAPTCHA in the browser window")
    print(f"{'='*60}\n")


def notify_browser_crash(error_msg: str = "") -> None:
    """Send browser crash notification."""
    title = "⚠️ Browser Crashed"
    message = "The browser has crashed. Attempting to recover..."
    if error_msg:
        message += f"\nError: {error_msg[:100]}"
    
    notify(title, message, NotificationType.BROWSER_CRASH, play_alert=True)


def notify_network_error(retry_count: int, max_retries: int) -> None:
    """Send network error notification."""
    title = "🌐 Network Error"
    message = f"Connection failed. Retry {retry_count}/{max_retries}..."
    
    notify(title, message, NotificationType.NETWORK_ERROR, play_alert=False)


def notify_success(company: str, role: str) -> None:
    """Send success notification after completing an application."""
    title = "✅ Application Ready"
    message = f"{role} at {company}\nReady for your review!"
    
    notify(title, message, NotificationType.SUCCESS, play_alert=True)


def notify_error(company: str, error_msg: str) -> None:
    """Send error notification."""
    title = "❌ Application Failed"
    message = f"{company}: {error_msg[:100]}"
    
    notify(title, message, NotificationType.ERROR, play_alert=True)


def generate_daily_summary(db) -> str:
    """
    Generate a daily summary of applications.
    
    Args:
        db: ApplicationDatabase instance
    
    Returns:
        Summary text
    """
    try:
        stats = db.get_statistics()
        by_status = stats.get("by_status", {})
        
        # Get today's applications
        from application_db import ApplicationStatus
        all_apps = db.get_all_applications(limit=1000)
        
        today = datetime.now().date()
        today_apps = [
            app for app in all_apps 
            if app.created_at.date() == today or 
               (app.updated_at and app.updated_at.date() == today)
        ]
        
        # Count today's stats
        today_completed = len([a for a in today_apps if a.status == ApplicationStatus.COMPLETED])
        today_submitted = len([a for a in today_apps if a.status == ApplicationStatus.SUBMITTED])
        today_failed = len([a for a in today_apps if a.status == ApplicationStatus.FAILED])
        
        summary_lines = [
            f"📊 Daily Summary - {today.strftime('%B %d, %Y')}",
            "",
            f"Today's Activity:",
            f"  • Completed: {today_completed}",
            f"  • Submitted: {today_submitted}",
            f"  • Failed: {today_failed}",
            "",
            f"All-Time Stats:",
            f"  • Total: {stats['total']}",
            f"  • Last 7 days: {stats['last_7_days']}",
            f"  • Queued: {by_status.get('queued', 0)}",
            f"  • Interviews: {by_status.get('interview', 0)}",
            f"  • Offers: {by_status.get('offer', 0)}",
        ]
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        return f"Could not generate summary: {e}"


def send_daily_summary(db) -> None:
    """Send the daily summary notification."""
    summary = generate_daily_summary(db)
    
    title = "📊 Daily Application Summary"
    # For desktop notification, use a shorter version
    short_summary = summary.split("\n\n")[1] if "\n\n" in summary else summary[:200]
    
    notify(title, short_summary, NotificationType.DAILY_SUMMARY, play_alert=True)
    
    # Also print full summary to console
    print(f"\n{'='*60}")
    print(summary)
    print(f"{'='*60}\n")


class DailySummaryScheduler:
    """
    Scheduler for daily summary notifications.
    Runs in a background thread.
    """
    
    def __init__(self, db, summary_time: str = "18:00"):
        """
        Initialize the scheduler.
        
        Args:
            db: ApplicationDatabase instance
            summary_time: Time to send summary (HH:MM format)
        """
        self.db = db
        self.summary_time = summary_time
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_sent_date: Optional[datetime] = None
    
    def start(self) -> None:
        """Start the scheduler in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"📅 Daily summary scheduler started (scheduled for {self.summary_time})")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("📅 Daily summary scheduler stopped")
    
    def _run_loop(self) -> None:
        """Main scheduler loop."""
        import time
        
        while self._running:
            try:
                now = datetime.now()
                
                # Parse summary time
                hour, minute = map(int, self.summary_time.split(":"))
                summary_datetime = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # Check if it's time to send and we haven't sent today
                if (now >= summary_datetime and 
                    (self._last_sent_date is None or self._last_sent_date.date() < now.date())):
                    
                    send_daily_summary(self.db)
                    self._last_sent_date = now
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except Exception as e:
                print(f"Daily summary scheduler error: {e}")
                time.sleep(60)
    
    def send_now(self) -> None:
        """Manually trigger the daily summary."""
        send_daily_summary(self.db)


# Global scheduler instance
_scheduler: Optional[DailySummaryScheduler] = None


def start_daily_summary_scheduler(db, summary_time: str = "18:00") -> DailySummaryScheduler:
    """
    Start the daily summary scheduler.
    
    Args:
        db: ApplicationDatabase instance
        summary_time: Time to send daily summary (HH:MM format)
    
    Returns:
        The scheduler instance
    """
    global _scheduler
    
    if _scheduler is not None:
        _scheduler.stop()
    
    _scheduler = DailySummaryScheduler(db, summary_time)
    _scheduler.start()
    return _scheduler


def stop_daily_summary_scheduler() -> None:
    """Stop the daily summary scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None


def get_scheduler() -> Optional[DailySummaryScheduler]:
    """Get the current scheduler instance."""
    return _scheduler


# Create sounds directory and placeholder files info
def setup_sounds_directory() -> str:
    """
    Create the sounds directory if it doesn't exist.
    Returns instructions for adding custom sounds.
    """
    sounds_dir = Path(__file__).parent / "sounds"
    sounds_dir.mkdir(exist_ok=True)
    
    readme_path = sounds_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# Sound Files for Notifications

Place your custom sound files here:

- `captcha_alert.wav` - Played when CAPTCHA is detected (urgent, attention-grabbing)
- `error.wav` - Played on errors or browser crashes
- `success.wav` - Played when application is completed

## Recommended Sources for Free Sounds:
- https://freesound.org/
- https://soundbible.com/
- https://mixkit.co/free-sound-effects/

## Tips:
- Keep files under 1MB for faster loading
- WAV format works best across platforms
- Short sounds (1-3 seconds) are ideal

If no custom sounds are provided, the system will use built-in beeps.
"""
        readme_path.write_text(readme_content)
    
    return str(sounds_dir)


# Initialize sounds directory on import
setup_sounds_directory()


if __name__ == "__main__":
    # Test notifications
    print("Testing notification system...")
    
    # Test desktop notification
    print("\n1. Testing desktop notification...")
    send_desktop_notification(
        "Test Notification",
        "This is a test message from the Job Application Agent",
        NotificationType.INFO
    )
    
    # Test sound
    print("\n2. Testing sound alert...")
    play_sound(NotificationType.CAPTCHA)
    
    # Test CAPTCHA notification
    import time
    time.sleep(2)
    print("\n3. Testing CAPTCHA notification...")
    notify_captcha("Test Company", "https://example.com/apply")
    
    # Test daily summary (if database exists)
    time.sleep(2)
    print("\n4. Testing daily summary...")
    try:
        from application_db import get_db
        db = get_db()
        summary = generate_daily_summary(db)
        print(summary)
    except Exception as e:
        print(f"Could not test daily summary: {e}")
    
    print("\n✅ Notification tests complete!")
