import streamlit as st
from crewai import Agent,Task,Crew,Process
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_MODEL_NAME"] = "llama-3.3-70b-versatile"
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

from crewai.llm import LLM


st.set_page_config(page_title="📊 CrewAI Market Report Generator", layout="centered")
st.title("📊 Multi-Agent Market Report Generator")

topic = st.text_input("Enter your report topic ")

if st.button("🚀 Run Agents and Generate Report"):
    with st.spinner("⚙️ Crew is working on your report..."):
        llm = LLM(
            model="llama3-8b-8192",
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        planner = Agent(
            role='Task Decomposer',
            goal=f'Break down a high-level objective on {topic} into detailed, actionable subtasks and delegate them effectively',
            backstory='You are a strategic planner with expertise in decomposing goals into structured subtasks for execution.',
            llm=llm
        )
        researcher = Agent(
            role='Researcher',
            goal=f'Conduct comprehensive research on {topic} including trends, key players, and market data',
            backstory='You are an expert market researcher specializing in artificial intelligence startups and funding trends.',
            llm=llm
        )
        analyst = Agent(
            role='Data Analyst',
            goal=f'Analyze market trends and identify key players in {topic} space',
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
            description=f'Break down the {topic} market report project into actionable steps',
            agent=planner,
            expected_output='Detailed task breakdown with assignments for researcher, analyst, and writer'
        )
        research_task = Task(
            description=f'Gather comprehensive information about current {topic} including funding, products, and founders',
            agent=researcher,
            expected_output=f'Bullet-point list of key findings about {topic} with sources',
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
            expected_output=f'A 3-5 page market report on {topic} with sections: Executive Summary, Market Overview, Key Players, Trends, and Future Outlook',
            context=[analysis_task]
        )
        crew = Crew(
            agents=[planner, researcher, analyst, writer],
            tasks=[planning_task, research_task, analysis_task, writing_task],
            process=Process.sequential,
            verbose=True,


        )
        final_report = crew.kickoff()
        st.success("Report Generation Complete!")
        st.subheader(f"📄 {topic} Market Report")
        st.markdown(final_report)
