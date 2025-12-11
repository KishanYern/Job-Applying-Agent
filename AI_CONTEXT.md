Project Context: Local Autonomous Job Agent

1. Project Goal

We are building a privacy-first, local desktop application that automates the process of applying to jobs.

The Problem: Applying to jobs is tedious.

The Solution: An AI Agent that runs locally on the user's machine, opens a visible browser window, navigates to a job URL, and autonomously fills out the application form using the user's resume data.

Key Constraint: All personal data (resume, phone number, address) must stay local. No external APIs (like OpenAI) should be used for inference. We use Ollama running locally.

2. The Tech Stack

Language: Python 3.10+

Frontend: Streamlit (for a simple drag-and-drop UI to upload resumes and paste URLs).

AI Engine: Ollama (running llama3.1 or llama3.2 models locally).

Browser Automation: Playwright (running in headless=False mode so the user can watch).

Agent Framework: browser-use (a Python library that connects LLMs to Playwright).

Data Handling: langchain-ollama for the LLM interface.

3. Architecture Overview

A. The Frontend (app.py)

A Streamlit app that serves as the control panel.

Inputs: 1. Text Input: Job Posting URL.
2. File Uploader: Resume (PDF).
3. Dropdown: Select Model (e.g., llama3.1, llama3.2).

Action: When the user clicks "Start Agent":

Save the uploaded PDF to a temporary local folder (e.g., data/current_resume.pdf).

Trigger the asynchronous backend agent logic.

B. The Backend Agent (apply_agent.py)

This script contains the core logic for the autonomous agent. It uses the browser-use library.

The Workflow:

Initialization: The agent initializes a Playwright browser instance.

Context Loading: It reads the user's "Knowledge Base" (a file named my_profile.md which contains their text-based resume, demographic info, and standard answers).

Navigation: It goes to the target Job URL provided by the frontend.

Reasoning Loop (The "Brain"):

The Agent analyzes the DOM (HTML) of the page.

It maps form fields (e.g., "First Name", "LinkedIn URL") to the data in my_profile.md.

It fills out the fields.

Advanced Reasoning (The "Researcher"):

If it encounters a "Salary Expectation" field: It opens a new tab, Googles the company+role average salary, extracts the number, closes the tab, and fills the form.

If it encounters "Why do you want to work here?": It researches the company's "About Us" page in a new tab to generate a relevant answer.

Human Handoff:

CAPTCHA: If the agent detects a CAPTCHA or gets stuck, it must PAUSE and play a sound or alert the user to intervene manually. It should not try to solve CAPTCHAs itself.

4. Specific File Requirements

my_profile.md (The Source of Truth)

This file already exists and contains the user's full profile. The agent must read this file into its system prompt context so it knows who it is applying as.

apply_agent.py (The Logic)

Must export a function: async def run_agent(job_url, resume_path, model_name).

Prompt Engineering: The System Prompt must explicitly tell the Llama model:

"You are an automated job applicant."

"Use the provided my_profile.md data to fill fields."

"Never submit the application without user review."

"If you see a salary field, open a new tab to research the market rate."

app.py (The UI)

Must handle the asyncio loop correctly to run the Playwright agent without freezing the Streamlit UI.

5. Development Guidelines for the Coder

Keep it Modular: Separate the UI code from the Agent logic.

Error Handling: If the browser crashes or the model hallucinates, fail gracefully and print the error to the Streamlit UI.

**Privacy