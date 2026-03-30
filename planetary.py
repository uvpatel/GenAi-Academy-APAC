import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

# --- Custom Tools ---

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's planetary inquiry to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}

# Configuring the Wikipedia Tool for general space knowledge
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# --- Agent Definitions ---

# 1. Planetary Researcher Agent
comprehensive_researcher = Agent(
    name="planetary_researcher",
    model=model_name,
    description="The primary researcher that can access both internal mission data and external cosmic knowledge from Wikipedia.",
    instruction="""
    You are an expert Astrophysicist. Your goal is to fully answer the user's PROMPT regarding celestial bodies.
    You have access to two tools:
    1. A tool for getting specific data about our CURRENT SPACE MISSIONS (vessel names, distances, landing sites).
    2. A tool for searching Wikipedia for general planetary knowledge (gravity, atmosphere, orbital period).

    First, analyze the user's PROMPT.
    - If the prompt can be answered by only one tool, use that tool.
    - If the prompt is complex and requires information from both mission logs AND Wikipedia,
        you MUST use both tools to gather all necessary information.
    - Synthesize the results from the tool(s) you use into preliminary research data.

    PROMPT:
    { PROMPT }
    """,
    tools=[
        wikipedia_tool
    ],
    output_key="research_data" # Stores findings for the next agent
)

# 2. Mission Briefing Formatter Agent
response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Synthesizes all planetary information into a professional mission briefing.",
    instruction="""
    You are the voice of Mission Control. Your task is to take the
    RESEARCH_DATA and present it to the user in a complete and educational answer.

    - First, present the specific information regarding our missions (like active rovers, satellites, or planned landings).
    - Then, add the interesting scientific facts from the general research (like what the planet is made of or its size).
    - If some information is missing, just present the best data you have available.
    - Use a professional, inspiring, and clear tone.

    RESEARCH_DATA:
    { research_data }
    """
)

# --- Workflow Setup ---

planet_explorer_workflow = SequentialAgent(
    name="planet_explorer_workflow",
    description="The main workflow for handling a user's request about a planet or moon.",
    sub_agents=[
        comprehensive_researcher, # Step 1: Gather astronomical data
        response_formatter,       # Step 2: Format the mission briefing
    ]
)

root_agent = Agent(
    name="mission_control_greeter",
    model=model_name,
    description="The main entry point for the Planet Explorer System.",
    instruction="""
    - Welcome the user to the Interstellar Exploration Bureau.
    - Let the user know you can help them learn about the planets, moons, and missions in our solar system.
    - When the user responds with a destination, use the 'add_prompt_to_state' tool to save their response.
    - After using the tool, transfer control to the 'planet_explorer_workflow' agent.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[planet_explorer_workflow]
)