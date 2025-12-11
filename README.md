# Local AI Job Application Agent

A privacy-focused, automated job application tool running entirely on local hardware. This agent uses a local LLM to navigate job portals, conduct real-time research on companies, and fill out application forms, leaving only sensitive actions (Logins/CAPTCHAs) to the human user.

## 🏗 Architecture

This project moves away from standard "resume rewriters" and functions as an **Autonomous Agent**.

* **Brain:** Llama 3.1 (8B) via [Ollama](https://ollama.com).
* **Hands:** [browser-use](https://github.com/browser-use/browser-use) (Playwright) for DOM interaction.
* **Orchestration:** LangChain.
* **Data Source:** A single Markdown file (`my_profile.md`) serving as the knowledge base.

### The "Researcher" Workflow
Unlike simple form-fillers, this agent handles open-ended questions (e.g., "Why do you want to work here?") via a multi-tab workflow:
1. **Detects** complex questions on the form.
2. **Opens** a new background tab.
3. **Googles** the company name found on the page.
4. **Reads** the company mission/values.
5. **Synthesizes** an answer linking the user's profile to the company's values.
6. **Closes** the tab and fills the form.

## 💻 Prerequisites

Since this runs locally, hardware requirements are strict.

* **OS:** Windows, macOS, or Linux.
* **RAM:** 16GB minimum recommended (8GB for the model + OS/Browser overhead).
* **Software:**
  * Python 3.10+
  * [Ollama](https://ollama.com/download) installed and running.

## 🛠 Installation

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/yourusername/local-job-agent.git](https://github.com/yourusername/local-job-agent.git)
   cd local-job-agent
   ```
Set up the Local Model Pull the Llama 3.1 8B model. This is required for the reasoning capabilities needed to manage browser tabs without hallucinating.

```bash

ollama pull llama3.1
Install Dependencies
```
```bash
pip install browser-use langchain-ollama playwright
playwright install
```