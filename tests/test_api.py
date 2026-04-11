import asyncio
import httpx
from dotenv import load_dotenv
from job_scraper import JobListing, research_company
import json

# Ensure environment variables are loaded
load_dotenv()

async def query_one_company():
    job = JobListing(
        company="Anthropic",
        role="Software Engineer - AI",
        location="San Francisco, CA",
        apply_url="https://www.anthropic.com/careers/software-engineer",
        source_repo="test"
    )
    async with httpx.AsyncClient() as client:
        print(f"Researching {job.company}...")
        data = await research_company(job, client)
        print("\n=== Research Data ===")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(query_one_company())
