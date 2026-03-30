import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

# Using LangChain's Python REPL tool to allow the agent to execute code
from langchain_experimental.tools import PythonREPLTool

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

# Added a fallback to avoid NoneType errors during build
model_name = os.getenv("MODEL", "gemini-1.5-flash")

# --- Custom Tools ---

def add_task_to_state(
    tool_context: ToolContext, task: str
) -> dict[str, str]:
    """Saves the user's coding task to the state."""
    tool_context.state["CODING_TASK"] = task
    logging.info(f"[State updated] Added to CODING_TASK: {task}")
    return {"status": "success"}

# Configuring the Python Execution Tool so the agent can test its code
python_execution_tool = LangchainTool(
    tool=PythonREPLTool()
)

# --- Agent Definitions ---

# 1. Senior Python Developer Agent
senior_developer = Agent(
    name="senior_python_developer",
    model=model_name,
    description="The primary developer that writes Python code and tests it using the REPL tool.",
    instruction="""
    You are an elite Python Software Engineer. Your goal is to solve the user's CODING_TASK.
    
    You have access to a Python REPL tool.
    
    First, analyze the user's CODING_TASK.
    1. Write the Python code to solve the problem.
    2. You MUST use the Python REPL tool to execute your code with basic test cases to ensure it works.
    3. If the tool returns an error, debug your code, fix it, and test it again.
    4. Once successful, compile the working code and the test results into your final output.

    CODING_TASK:
    {CODING_TASK}
    """,
    tools=[
        python_execution_tool
    ],
    output_key="tested_code_data" # Stores the raw working code for the reviewer
)

# 2. Tech Lead Reviewer Agent
tech_lead_reviewer = Agent(
    name="tech_lead_reviewer",
    model=model_name,
    description="Reviews the code, ensures PEP8 compliance, and formats the final output.",
    instruction="""
    You are a strict but helpful Tech Lead. Your task is to take the
    TESTED_CODE_DATA from the developer and present it to the user.

    - Ensure the code follows Python best practices (PEP8).
    - Add meaningful docstrings and comments if the developer missed them.
    - Provide a brief, clear explanation of how the code works and its time/space complexity if applicable.
    - Format the final output professionally using Markdown code blocks.

    TESTED_CODE_DATA:
    {tested_code_data}
    """
)

# --- Workflow Setup ---

python_dev_workflow = SequentialAgent(
    name="python_dev_workflow",
    description="The main workflow for writing, testing, and reviewing Python code.",
    sub_agents=[
        senior_developer,   # Step 1: Write and test the code
        tech_lead_reviewer, # Step 2: Review, document, and format the response
    ]
)

# CRITICAL: Renamed from root_agent to agent for ADK deployment entry point
agent = Agent(
    name="engineering_manager_greeter",
    model=model_name,
    description="The main entry point for the AI Engineering Department.",
    instruction="""
    - Welcome the user to the AI Engineering Department.
    - Let the user know you can help them write, debug, and optimize Python code.
    - When the user responds with a coding task, use the 'add_task_to_state' tool to save their request.
    - After using the tool, transfer control to the 'python_dev_workflow' agent to complete the task.
    """,
    tools=[add_task_to_state],
    sub_agents=[python_dev_workflow]
)