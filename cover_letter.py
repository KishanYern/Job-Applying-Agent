"""
Cover Letter Generator - Creates tailored cover letters using the Groq API (Llama-3.3-70B).
Phase 1 pre-generates all cover letters during discovery so Phase 2 never writes them inline.
"""

import os
from dataclasses import dataclass
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class JobDetails:
    """Job information for cover letter personalization."""
    company: str
    role: str
    job_description: Optional[str] = None
    location: Optional[str] = None
    company_about: Optional[str] = None  # Company mission/values from Phase 1 research


COVER_LETTER_SYSTEM_PROMPT = """Write a short cover letter (under 300 words, 3 paragraphs).
OUTPUT ONLY THE LETTER TEXT. DO NOT INCLUDE PREAMBLES, INTRODUCTIONS, OR OUTROS.

## BANNED WORDS/PHRASES - Never use these:
- "I am writing to express"
- "I am excited" or "I am thrilled"  
- "leverage" or "utilize" (say "use" instead)
- "passionate about"
- "I believe I would be a great fit"
- the dash character "—"

## REQUIRED STYLE:
- Use contractions: write "I'm" not "I am", write "I've" not "I have"
- Keep sentences short
- Be specific about experience, not vague

## EXAMPLE:

Dear Acme Team,

I came across the Backend Engineer role and wanted to reach out. I've been building APIs and data pipelines for the past two years at a fintech startup, and your focus on scalable infrastructure is right up my alley.

At my current job, I rebuilt our payment processing system to handle 5x more traffic. I also set up the monitoring stack that cut our incident response time in half. I'm comfortable with Python, PostgreSQL, and AWS, which I saw in your job posting.

I'd love to chat more about how I could help out. Thanks for considering my application.

Best,
Kishan

## FORMAT:
- Start with "Dear [Company] Team," 
- End with "Best," and the candidate's name
- No addresses or dates
- Use real info from the profile only"""


def _get_llm() -> ChatGroq:
    """Return a ChatGroq instance using GROQ_API_KEY from environment."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.7,
    )


def build_cover_letter_prompt(profile_content: str, job: JobDetails) -> str:
    """Build the prompt for cover letter generation."""
    prompt = f"""Generate a tailored cover letter for the following job application.

## CANDIDATE PROFILE
{profile_content}

## JOB DETAILS
- **Company:** {job.company}
- **Position:** {job.role}
"""

    if job.location:
        prompt += f"- **Location:** {job.location}\n"

    if job.job_description:
        prompt += f"\n## JOB DESCRIPTION\n{job.job_description}\n"

    if job.company_about:
        prompt += f"\n## ABOUT THE COMPANY\n{job.company_about}\n"

    prompt += """
## INSTRUCTIONS
Write a 3-paragraph cover letter under 300 words.

RULES:
1. Use contractions (I'm, I've, I'd) - never write "I am" or "I have"
2. Never say "I am excited", "leverage", "utilize", or "passionate about"
3. Mention 2 specific things from the profile that match the job
4. End with "I'd love to chat more" or similar

Keep it short and natural."""

    return prompt


async def generate_cover_letter_async(
    profile_content: str,
    job: JobDetails,
) -> str:
    """
    Generate a tailored cover letter using Groq (Llama-3.3-70B).

    Args:
        profile_content: The candidate's profile from my_profile.md
        job: Job details for personalization

    Returns:
        Generated cover letter text
    """
    llm = _get_llm()

    messages = [
        SystemMessage(content=COVER_LETTER_SYSTEM_PROMPT),
        HumanMessage(content=build_cover_letter_prompt(profile_content, job)),
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()


def generate_cover_letter_sync(
    profile_content: str,
    job: JobDetails,
) -> str:
    """
    Synchronous cover letter generation using Groq (Llama-3.3-70B).

    Args:
        profile_content: The candidate's profile from my_profile.md
        job: Job details for personalization

    Returns:
        Generated cover letter text
    """
    llm = _get_llm()

    messages = [
        SystemMessage(content=COVER_LETTER_SYSTEM_PROMPT),
        HumanMessage(content=build_cover_letter_prompt(profile_content, job)),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


# For standalone testing
if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    profile_path = Path(__file__).parent / "my_profile.md"
    if profile_path.exists():
        profile = profile_path.read_text(encoding="utf-8")
    else:
        print("my_profile.md not found - using sample profile")
        profile = """
Name: John Doe
Email: john@example.com
Phone: 555-123-4567

Experience:
- Software Engineer at Tech Corp (2022-Present)
  - Built microservices using Python and Go
  - Reduced API latency by 40% through optimization
  - Led migration to Kubernetes

Education:
- B.S. Computer Science, State University (2022)

Skills: Python, Go, Kubernetes, AWS, PostgreSQL, React
"""

    test_job = JobDetails(
        company="Acme Tech",
        role="Senior Software Engineer",
        job_description="Looking for an experienced engineer to build scalable backend systems. Must have experience with Python, cloud infrastructure, and distributed systems.",
        location="Remote",
    )

    print("Generating cover letter...")
    print("=" * 60)
    cover_letter = asyncio.run(generate_cover_letter_async(profile, test_job))
    print(cover_letter)
    print("=" * 60)
