"""
Application Tracking Database - Local SQLite storage for job applications.
Keeps all data offline and private on the user's machine.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class ApplicationStatus(Enum):
    """Status of a job application."""
    QUEUED = "queued"           # In queue, not yet started
    IN_PROGRESS = "in_progress" # Agent is currently filling it out
    COMPLETED = "completed"     # Successfully filled, awaiting review
    SUBMITTED = "submitted"     # User confirmed submission
    FAILED = "failed"           # Agent encountered an error
    SKIPPED = "skipped"         # User skipped this application
    REJECTED = "rejected"       # Company rejected
    INTERVIEW = "interview"     # Got an interview
    OFFER = "offer"             # Received an offer


@dataclass
class Application:
    """Represents a job application record."""
    id: Optional[int]
    company: str
    role: str
    job_url: str
    location: Optional[str]
    status: ApplicationStatus
    source: Optional[str]       # Where we found the job (e.g., "SimplifyJobs/Summer2025-Internships")
    resume_used: Optional[str]  # Path to resume file used
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    applied_at: Optional[datetime]  # When actually submitted
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Application":
        """Create an Application from a database row."""
        return cls(
            id=row["id"],
            company=row["company"],
            role=row["role"],
            job_url=row["job_url"],
            location=row["location"],
            status=ApplicationStatus(row["status"]),
            source=row["source"],
            resume_used=row["resume_used"],
            notes=row["notes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            applied_at=datetime.fromisoformat(row["applied_at"]) if row["applied_at"] else None,
        )


@dataclass
class SavedAnswer:
    """Represents a saved answer for common application questions."""
    id: Optional[int]
    question_pattern: str  # Regex or keyword pattern to match questions
    answer: str
    company: Optional[str]  # If company-specific, otherwise None for generic
    created_at: datetime
    used_count: int


class ApplicationDatabase:
    """
    SQLite database manager for tracking job applications.
    All data stays local - nothing touches the internet.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file. 
                     Defaults to 'data/applications.db' in the project directory.
        """
        if db_path is None:
            db_path = str(Path(__file__).parent / "data" / "applications.db")
        
        self.db_path = db_path
        
        # Ensure the data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Applications table - core tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                job_url TEXT NOT NULL UNIQUE,
                location TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                source TEXT,
                resume_used TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                applied_at TEXT
            )
        """)
        
        # Index on job_url for fast duplicate checking
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_applications_job_url 
            ON applications(job_url)
        """)
        
        # Index on status for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_applications_status 
            ON applications(status)
        """)
        
        # Index on company for grouping
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_applications_company 
            ON applications(company)
        """)
        
        # Saved answers table - for answer bank feature
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_pattern TEXT NOT NULL,
                answer TEXT NOT NULL,
                company TEXT,
                created_at TEXT NOT NULL,
                used_count INTEGER DEFAULT 0
            )
        """)
        
        # Companies table - cache company research
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                website TEXT,
                about_summary TEXT,
                values_summary TEXT,
                salary_data TEXT,
                last_researched TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ==================== Application CRUD ====================
    
    def add_application(
        self,
        company: str,
        role: str,
        job_url: str,
        location: Optional[str] = None,
        status: ApplicationStatus = ApplicationStatus.QUEUED,
        source: Optional[str] = None,
        resume_used: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Optional[int]:
        """
        Add a new application to the database.
        
        Returns:
            The application ID if successful, None if duplicate URL exists.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT INTO applications 
                (company, role, job_url, location, status, source, resume_used, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (company, role, job_url, location, status.value, source, resume_used, notes, now, now))
            
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Duplicate URL
            return None
        finally:
            conn.close()
    
    def get_application(self, application_id: int) -> Optional[Application]:
        """Get an application by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM applications WHERE id = ?", (application_id,))
        row = cursor.fetchone()
        conn.close()
        
        return Application.from_row(row) if row else None
    
    def get_application_by_url(self, job_url: str) -> Optional[Application]:
        """Get an application by job URL (for duplicate checking)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM applications WHERE job_url = ?", (job_url,))
        row = cursor.fetchone()
        conn.close()
        
        return Application.from_row(row) if row else None
    
    def is_duplicate(self, job_url: str) -> bool:
        """Check if a job URL has already been applied to."""
        return self.get_application_by_url(job_url) is not None
    
    def update_status(
        self, 
        application_id: int, 
        status: ApplicationStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Update the status of an application."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        applied_at = now if status == ApplicationStatus.SUBMITTED else None
        
        if notes:
            cursor.execute("""
                UPDATE applications 
                SET status = ?, notes = ?, updated_at = ?, applied_at = COALESCE(applied_at, ?)
                WHERE id = ?
            """, (status.value, notes, now, applied_at, application_id))
        else:
            cursor.execute("""
                UPDATE applications 
                SET status = ?, updated_at = ?, applied_at = COALESCE(applied_at, ?)
                WHERE id = ?
            """, (status.value, now, applied_at, application_id))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    
    def get_all_applications(
        self, 
        status: Optional[ApplicationStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Application]:
        """Get all applications, optionally filtered by status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT * FROM applications 
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (status.value, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM applications 
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Application.from_row(row) for row in rows]
    
    def get_statistics(self) -> dict:
        """Get application statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM applications")
        stats["total"] = cursor.fetchone()[0]
        
        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM applications 
            GROUP BY status
        """)
        stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        # Recent applications (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM applications 
            WHERE created_at > datetime('now', '-7 days')
        """)
        stats["last_7_days"] = cursor.fetchone()[0]
        
        # Unique companies
        cursor.execute("SELECT COUNT(DISTINCT company) FROM applications")
        stats["unique_companies"] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def search_applications(self, query: str) -> list[Application]:
        """Search applications by company or role."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT * FROM applications 
            WHERE company LIKE ? OR role LIKE ?
            ORDER BY updated_at DESC
            LIMIT 50
        """, (search_term, search_term))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Application.from_row(row) for row in rows]
    
    def delete_application(self, application_id: int) -> bool:
        """Delete an application by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM applications WHERE id = ?", (application_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    
    # ==================== Saved Answers ====================
    
    def save_answer(
        self,
        question_pattern: str,
        answer: str,
        company: Optional[str] = None
    ) -> int:
        """Save an answer for future use."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO saved_answers (question_pattern, answer, company, created_at)
            VALUES (?, ?, ?, ?)
        """, (question_pattern, answer, company, now))
        
        conn.commit()
        answer_id = cursor.lastrowid
        conn.close()
        return answer_id
    
    def find_answer(self, question: str, company: Optional[str] = None) -> Optional[str]:
        """Find a saved answer matching the question."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # First try company-specific answers
        if company:
            cursor.execute("""
                SELECT answer FROM saved_answers 
                WHERE company = ? AND ? LIKE '%' || question_pattern || '%'
                ORDER BY used_count DESC
                LIMIT 1
            """, (company, question.lower()))
            row = cursor.fetchone()
            if row:
                conn.close()
                return row["answer"]
        
        # Fall back to generic answers
        cursor.execute("""
            SELECT answer FROM saved_answers 
            WHERE company IS NULL AND ? LIKE '%' || question_pattern || '%'
            ORDER BY used_count DESC
            LIMIT 1
        """, (question.lower(),))
        row = cursor.fetchone()
        conn.close()
        
        return row["answer"] if row else None
    
    # ==================== Company Cache ====================
    
    def cache_company_info(
        self,
        name: str,
        website: Optional[str] = None,
        about_summary: Optional[str] = None,
        values_summary: Optional[str] = None,
        salary_data: Optional[str] = None
    ) -> None:
        """Cache company research data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO companies (name, website, about_summary, values_summary, salary_data, last_researched, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                website = COALESCE(excluded.website, website),
                about_summary = COALESCE(excluded.about_summary, about_summary),
                values_summary = COALESCE(excluded.values_summary, values_summary),
                salary_data = COALESCE(excluded.salary_data, salary_data),
                last_researched = excluded.last_researched
        """, (name, website, about_summary, values_summary, salary_data, now, now))
        
        conn.commit()
        conn.close()
    
    def get_company_info(self, name: str) -> Optional[dict]:
        """Get cached company info."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM companies WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None


# Singleton instance for easy import
_db_instance: Optional[ApplicationDatabase] = None


def get_db() -> ApplicationDatabase:
    """Get the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ApplicationDatabase()
    return _db_instance


# CLI for testing
if __name__ == "__main__":
    import sys
    
    db = get_db()
    print(f"Database initialized at: {db.db_path}")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "stats":
            stats = db.get_statistics()
            print("\n📊 Application Statistics:")
            print(f"   Total applications: {stats['total']}")
            print(f"   Last 7 days: {stats['last_7_days']}")
            print(f"   Unique companies: {stats['unique_companies']}")
            print("\n   By status:")
            for status, count in stats.get("by_status", {}).items():
                print(f"      {status}: {count}")
        
        elif command == "list":
            apps = db.get_all_applications(limit=20)
            print(f"\n📋 Recent Applications ({len(apps)}):\n")
            for app in apps:
                status_emoji = {
                    "queued": "⏳", "in_progress": "🔄", "completed": "✅",
                    "submitted": "📤", "failed": "❌", "rejected": "👎",
                    "interview": "🎯", "offer": "🎉"
                }.get(app.status.value, "❓")
                print(f"   {status_emoji} [{app.status.value}] {app.company} - {app.role}")
                print(f"      URL: {app.job_url[:60]}...")
                print()
        
        elif command == "test":
            # Add a test application
            app_id = db.add_application(
                company="Test Company",
                role="Software Engineer",
                job_url=f"https://example.com/jobs/test-{datetime.now().timestamp()}",
                location="Remote",
                source="manual"
            )
            if app_id:
                print(f"✅ Test application added with ID: {app_id}")
            else:
                print("❌ Failed to add (duplicate?)")
    else:
        print("\nUsage:")
        print("  python application_db.py stats  - Show statistics")
        print("  python application_db.py list   - List recent applications")
        print("  python application_db.py test   - Add a test application")
