from crewai import Agent,Task,Crew,Process
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_MODEL_NAME"] = "llama3-8b-8192"
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

from crewai.llm import LLM


llm = LLM(
    model="llama3-8b-8192",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

planner = Agent(
    role='Task Decomposer',
    goal='Break down a high-level objective into detailed, actionable subtasks and delegate them effectively',
    backstory='You are a strategic planner with expertise in decomposing goals into structured subtasks for execution.',
    llm=llm
)
researcher = Agent(
    role='Researcher',
    goal='Conduct comprehensive research on LLM startups including trends, key players, and market data',
    backstory='You are an expert market researcher specializing in artificial intelligence startups and funding trends.',
    llm=llm
)
analyst = Agent(
    role='Data Analyst',
    goal='Analyze market trends and identify key players in LLM startup space',
    backstory='Data scientist with experience in startup valuation and market analysis',
    llm=llm
)
writer = Agent(
    role='Writer',
    goal='Write a clear and concise market report based on research findings',
    backstory='You are a skilled technical writer with experience in creating insightful reports on technology sectors.',
    llm=llm
)
planning_task = Task(
    description='Break down the LLM startup market report project into actionable steps',
    agent=planner,
    expected_output='Detailed task breakdown with assignments for researcher, analyst, and writer'
)
research_task = Task(
    description='Gather comprehensive information about current LLM startups including funding, products, and founders',
    agent=researcher,
    expected_output='Bullet-point list of key findings about LLM startups with sources',
    context=[planning_task]
)
analysis_task = Task(
    description='Analyze the research data to identify market trends, key players, and growth opportunities',
    agent=analyst,
    expected_output='Structured analysis of market trends with visualizable data points',
    context=[research_task]
)
writing_task = Task(
    description='Compile the research and analysis into a professional market report',
    agent=writer,
    expected_output='A 3-5 page market report on LLM startups with sections: Executive Summary, Market Overview, Key Players, Trends, and Future Outlook',
    context=[analysis_task]
)
crew = Crew(
    agents=[planner, researcher, analyst, writer],
    tasks=[planning_task, research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True,


)
crew.kickoff()
