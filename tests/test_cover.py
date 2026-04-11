import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from cover_letter import generate_cover_letter_async, JobDetails

profile_path = Path("my_profile.md")
if profile_path.exists():
    profile = profile_path.read_text(encoding="utf-8")
else:
    profile = "Sample placeholder..."

test_job = JobDetails(
    company="Anthropic",
    role="Software Engineer",
    job_description="Seeking backend software engineer.",
    location="San Francisco",
)

cover_letter = asyncio.run(generate_cover_letter_async(profile, test_job))
print("\n=== GENERATED COVER LETTER ===")
print(cover_letter)
print("Word Count:", len(cover_letter.split()))
