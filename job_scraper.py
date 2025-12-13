"""
Job URL Scraper - Fetches job listings from community-maintained GitHub repositories.
Targets software engineering, AI, and data science roles.
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import httpx


@dataclass
class JobListing:
    """Represents a single job listing."""
    company: str
    role: str
    location: str
    apply_url: str
    source_repo: str
    date_posted: Optional[str] = None
    is_open: bool = True
    
    def matches_filters(
        self, 
        keywords: list[str], 
        locations: list[str] | None = None,
        skip_location_filter: bool = False
    ) -> bool:
        """Check if job matches keyword and location filters."""
        role_lower = self.role.lower()
        
        # Check if any keyword matches the role
        keyword_match = any(kw.lower() in role_lower for kw in keywords) if keywords else True
        
        # Skip location filter if requested (e.g., for internships where user can relocate)
        if skip_location_filter:
            return keyword_match
        
        # Check location filter if provided
        if locations:
            location_lower = self.location.lower()
            location_match = any(loc.lower() in location_lower for loc in locations) or "remote" in location_lower
        else:
            location_match = True
            
        return keyword_match and location_match


@dataclass
class RepoConfig:
    """Configuration for a GitHub repository to scrape."""
    owner: str
    repo: str
    branch: str = "main"
    readme_path: str = "README.md"
    job_type: str = "internship"  # "internship" or "new-grad"
    
    @property
    def raw_url(self) -> str:
        """Get the raw GitHub URL for the README."""
        return f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}/{self.readme_path}"
    
    @property
    def display_name(self) -> str:
        return f"{self.owner}/{self.repo}"


# Pre-configured repositories for job hunting
DEFAULT_REPOS: list[RepoConfig] = [
    # SimplifyJobs - Gold standard lists (Summer 2026)
    RepoConfig(
        owner="SimplifyJobs",
        repo="Summer2026-Internships",
        branch="dev",
        job_type="internship"
    ),
    RepoConfig(
        owner="SimplifyJobs",
        repo="New-Grad-Positions",
        branch="dev",
        job_type="new-grad"
    ),
    # Pitt CSC - Now redirects to SimplifyJobs (Summer 2026)
    RepoConfig(
        owner="pittcsc",
        repo="Summer2026-Internships",
        branch="dev",
        job_type="internship"
    ),
    # Ouckah - Active list with markdown tables (Summer 2025 still active for some)
    RepoConfig(
        owner="Ouckah",
        repo="Summer2025-Internships",
        branch="dev",
        job_type="internship"
    ),
]

# Keywords for filtering relevant roles (Software Engineering, AI, Data Science)
SWE_AI_DS_KEYWORDS = [
    # Software Engineering
    "software", "engineer", "developer", "swe", "backend", "frontend", "fullstack",
    "full-stack", "full stack", "web dev", "mobile", "ios", "android", "platform",
    # AI/ML
    "machine learning", "ml", "artificial intelligence", "ai", "deep learning",
    "nlp", "natural language", "computer vision", "cv", "robotics",
    # Data Science
    "data scientist", "data science", "data analyst", "analytics", "data engineer",
    "quantitative", "quant", "research scientist", "applied scientist"
]


class JobScraper:
    """Scrapes job listings from GitHub repositories."""
    
    def __init__(self, repos: list[RepoConfig] | None = None):
        self.repos = repos or DEFAULT_REPOS
        self.client: httpx.AsyncClient | None = None
        
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def fetch_readme(self, repo: RepoConfig) -> str | None:
        """Fetch the raw README content from a repository."""
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            response = await self.client.get(repo.raw_url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            print(f"[ERROR] Failed to fetch {repo.display_name}: {e}")
            return None
    
    def parse_markdown_table(self, content: str, repo: RepoConfig) -> list[JobListing]:
        """Parse markdown table rows into JobListing objects."""
        listings: list[JobListing] = []
        
        # Match markdown table rows: | Company | Role | Location | Link | ... |
        # These repos typically have tables with apply links
        table_row_pattern = r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*'
        
        lines = content.split('\n')
        in_table = False
        
        for line in lines:
            line = line.strip()
            
            # Skip header separator rows
            if re.match(r'^\|[\s\-:|]+\|$', line):
                in_table = True
                continue
            
            # Skip if not a table row
            if not line.startswith('|'):
                in_table = False
                continue
                
            if not in_table:
                continue
            
            # Parse the row
            match = re.match(table_row_pattern, line)
            if not match:
                continue
            
            company = self._clean_cell(match.group(1))
            role = self._clean_cell(match.group(2))
            location = self._clean_cell(match.group(3))
            link_cell = match.group(4)
            
            # Skip closed positions (marked with strikethrough ~~)
            if '~~' in company or '🔒' in line.lower() or 'closed' in line.lower():
                continue
            
            # Extract URL from the link cell
            apply_url = self._extract_url(link_cell)
            if not apply_url:
                continue
            
            # Skip non-job URLs
            if any(skip in apply_url.lower() for skip in ['github.com', 'linkedin.com/company', 'twitter.com']):
                continue
            
            listing = JobListing(
                company=company,
                role=role,
                location=location,
                apply_url=apply_url,
                source_repo=repo.display_name
            )
            listings.append(listing)
        
        return listings
    
    def parse_html_table(self, content: str, repo: RepoConfig) -> list[JobListing]:
        """Parse HTML table rows (used by SimplifyJobs/pittcsc repos)."""
        listings: list[JobListing] = []
        
        # Match HTML table rows: <tr><td>...</td><td>...</td>...</tr>
        # Handle <td> with or without style attributes
        # SimplifyJobs format: Company | Role | Location | Application Link | Age
        row_pattern = r'<tr>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>'
        
        for match in re.finditer(row_pattern, content, re.DOTALL | re.IGNORECASE):
            company_cell = match.group(1)
            role_cell = match.group(2)
            location_cell = match.group(3)
            link_cell = match.group(4)
            
            # Skip continuation rows (↳ symbol)
            if '↳' in company_cell:
                # For continuation rows, company is inherited - skip for now
                continue
            
            company = self._clean_cell(company_cell)
            role = self._clean_cell(role_cell)
            location = self._clean_cell(location_cell)
            
            # Skip closed positions (🔒 emoji in the row)
            full_row = match.group(0)
            if '🔒' in full_row:
                continue
            
            # Skip empty companies
            if not company or company == '↳':
                continue
            
            # Extract URL from the link cell (look for actual job application link, not Simplify)
            apply_url = self._extract_job_url_from_html(link_cell)
            if not apply_url:
                continue
            
            # Skip non-job URLs
            if any(skip in apply_url.lower() for skip in ['github.com', 'linkedin.com/company', 'twitter.com', 'simplify.jobs/p/']):
                continue
            
            listing = JobListing(
                company=company,
                role=role,
                location=location,
                apply_url=apply_url,
                source_repo=repo.display_name
            )
            listings.append(listing)
        
        return listings
    
    def _extract_job_url_from_html(self, cell: str) -> str | None:
        """Extract the actual job URL from HTML (not Simplify redirect links)."""
        # Look for links that are NOT simplify.jobs (those are redirects)
        # Pattern: <a href="https://actual-company-url...">
        all_urls = re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', cell, re.IGNORECASE)
        
        for url in all_urls:
            # Skip Simplify redirect links, we want the actual company URL
            if 'simplify.jobs' not in url.lower() and url.startswith('http'):
                return url.split('?utm_source=')[0]  # Clean up tracking params but keep the base
        
        # Fallback: return first URL if no non-Simplify found
        if all_urls:
            return all_urls[0]
        
        return None
    
    def _clean_cell(self, cell: str) -> str:
        """Clean a markdown table cell."""
        # Remove markdown links but keep text
        cell = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cell)
        # Remove other markdown formatting
        cell = re.sub(r'[*_~`]', '', cell)
        # Remove HTML tags
        cell = re.sub(r'<[^>]+>', '', cell)
        # Remove emojis and non-ASCII characters
        cell = cell.encode('ascii', 'ignore').decode('ascii')
        return cell.strip()
    
    def _extract_url(self, cell: str) -> str | None:
        """Extract URL from a markdown cell."""
        # Try markdown link format: [text](url)
        md_match = re.search(r'\[([^\]]*)\]\(([^)]+)\)', cell)
        if md_match:
            return md_match.group(2).strip()
        
        # Try raw URL
        url_match = re.search(r'https?://[^\s<>"\')]+', cell)
        if url_match:
            return url_match.group(0).strip()
        
        return None
    
    async def scrape_repo(self, repo: RepoConfig) -> list[JobListing]:
        """Scrape a single repository for job listings."""
        content = await self.fetch_readme(repo)
        if not content:
            return []
        
        # Try markdown tables first (Ouckah style)
        listings = self.parse_markdown_table(content, repo)
        
        # If no markdown tables found, try HTML tables (SimplifyJobs/pittcsc style)
        # Check for '<table' to handle tables with attributes like <table style="...">
        if not listings and '<table' in content.lower():
            listings = self.parse_html_table(content, repo)
        
        print(f"[INFO] Found {len(listings)} listings from {repo.display_name}")
        return listings
    
    async def scrape_all(
        self,
        keywords: list[str] | None = None,
        locations: list[str] | None = None,
        job_type: str | None = None
    ) -> list[JobListing]:
        """
        Scrape all configured repositories.
        
        Args:
            keywords: Filter roles by keywords (default: SWE/AI/DS roles)
            locations: Filter by locations (e.g., ["Houston", "Remote", "Texas"])
            job_type: Filter by job type ("internship" or "new-grad")
        
        Returns:
            List of matching job listings.
        """
        if keywords is None:
            keywords = SWE_AI_DS_KEYWORDS
        
        # Filter repos by job type if specified
        repos_to_scrape = self.repos
        if job_type:
            repos_to_scrape = [r for r in self.repos if r.job_type == job_type]
        
        # Scrape all repos concurrently
        tasks = [self.scrape_repo(repo) for repo in repos_to_scrape]
        results = await asyncio.gather(*tasks)
        
        # Flatten and deduplicate by URL
        all_listings: list[JobListing] = []
        seen_urls: set[str] = set()
        
        # For internships, skip location filter (user can relocate anywhere)
        skip_location = job_type == "internship"
        
        for listings in results:
            for listing in listings:
                if listing.apply_url not in seen_urls:
                    # Apply filters
                    if listing.matches_filters(keywords, locations, skip_location_filter=skip_location):
                        all_listings.append(listing)
                        seen_urls.add(listing.apply_url)
        
        print(f"[INFO] Total unique listings after filtering: {len(all_listings)}")
        return all_listings


async def get_job_urls(
    keywords: list[str] | None = None,
    locations: list[str] | None = None,
    job_type: str | None = None,
    repos: list[RepoConfig] | None = None
) -> list[JobListing]:
    """
    Convenience function to fetch job URLs.
    
    Args:
        keywords: Role keywords to filter (default: SWE/AI/DS keywords)
        locations: Location filters (e.g., ["Remote", "Houston"])
        job_type: "internship" or "new-grad" (default: both)
        repos: Custom repo configurations (default: standard job boards)
    
    Returns:
        List of JobListing objects with apply URLs.
    
    Example:
        jobs = await get_job_urls(
            keywords=["machine learning", "data science"],
            locations=["Remote", "Texas"],
            job_type="internship"
        )
        for job in jobs:
            print(f"{job.company} - {job.role}: {job.apply_url}")
    """
    async with JobScraper(repos) as scraper:
        return await scraper.scrape_all(keywords, locations, job_type)


def get_job_urls_sync(
    keywords: list[str] | None = None,
    locations: list[str] | None = None,
    job_type: str | None = None
) -> list[JobListing]:
    """Synchronous wrapper for get_job_urls."""
    return asyncio.run(get_job_urls(keywords, locations, job_type))


# CLI interface
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Job URL Scraper - Fetching from GitHub repositories")
    print("=" * 60)
    
    # Parse command line args
    job_type_filter = None
    location_filter = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ("internship", "new-grad"):
            job_type_filter = sys.argv[1]
    if len(sys.argv) > 2:
        location_filter = sys.argv[2].split(",")
    
    print(f"\nJob Type: {job_type_filter or 'all'}")
    if job_type_filter == "internship":
        print("Locations: any (internships - willing to relocate)")
    else:
        print(f"Locations: {location_filter or 'all'}")
    print(f"Keywords: SWE/AI/Data Science roles")
    print("-" * 60)
    
    jobs = get_job_urls_sync(
        job_type=job_type_filter,
        locations=location_filter
    )
    
    if not jobs:
        print("\nNo jobs found matching your criteria.")
        sys.exit(0)
    
    # Group by company
    print(f"\nFound {len(jobs)} matching positions:\n")
    
    for i, job in enumerate(jobs[:50], 1):  # Show first 50
        print(f"{i:3}. {job.company}")
        print(f"     Role: {job.role}")
        print(f"     Location: {job.location}")
        print(f"     URL: {job.apply_url}")
        print(f"     Source: {job.source_repo}")
        print()
    
    if len(jobs) > 50:
        print(f"... and {len(jobs) - 50} more positions.")
    
    # Export option
    print("-" * 60)
    print("TIP: To export: python job_scraper.py > jobs.txt")
    print("TIP: Filter by type: python job_scraper.py internship")
    print("TIP: Filter by location: python job_scraper.py new-grad Remote,Texas")
