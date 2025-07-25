import streamlit as st
import sys
sys.setrecursionlimit(1500)
from crewai import Agent,Task,Crew
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_MODEL_NAME"] = "llama-3.3-70b-versatile"

from crewai.llm import LLM

llm = LLM(
    model="llama-3.3-70b-versatile",
    api_key="GROQ_API_KEY",
    base_url="https://api.groq.com/openai/v1",

)

st.set_page_config(page_title="🧠 AI Blog Post Generator", layout="centered")
st.title("🧠 AI Blog Post Generator")

topic = st.text_input("Enter your topic to generate the blog")

if st.button("Generate Blog"):
    if not topic:
        st.warning("Please enter a topic before generating.")
    else:
        with st.spinner("⚙️ Crew is working on your blog..."):

            planner = Agent(
                role="Content Planner",
                goal=f"Plan engaging and factually accurate content on {topic}",
                backstory=(
                    f"You are working on planning a blog article about {topic}. "
                    "You collect information that helps the audience learn something "
                    "and make informed decisions. Your work is the basis for the "
                    "content writer to write an article on this topic."
                ),
                allow_delegation=False,
                llm=llm
            )

            writer = Agent(
                role="Content Writer",
                goal=f"Write insightful and factually accurate opinion piece about the topic: {topic}",
                backstory=(
                    f"You're writing a new opinion piece about the topic: {topic}. "
                    "You base your writing on the work of the Content Planner, who provides an outline "
                    "and relevant context about the topic. You follow the main objectives and "
                    "direction of the outline, as provided by the Content Planner. You also provide "
                    "objective and impartial insights and back them up with information provided by the Content Planner. "
                    "You acknowledge in your opinion piece when your statements are opinions as opposed to objective statements."
                ),
                allow_delegation=False,
                llm=llm
            )

            editor = Agent(
                role="Editor",
                goal="Edit a given blog post to align with "
                     "the writing style of the organization. ",
                backstory="You are an editor who receives a blog post "
                          "from the Content Writer. "
                          "Your goal is to review the blog post "
                          "to ensure that it follows journalistic best practices,"
                          "provides balanced viewpoints "
                          "when providing opinions or assertions, "
                          "and also avoids major controversial topics "
                          "or opinions when possible.",
                allow_delegation=False,
                llm=llm

            )
            plan = Task(
                description=(
                    f"1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
                    "2. Identify the target audience, considering their interests and pain points.\n"
                    "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
                    "4. Include SEO keywords and relevant data or sources."
                ),
                expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
                agent=planner,
            )

            write = Task(
                description=(
                    f"1. Use the content plan to craft a compelling blog post on {topic}.\n"
                    "2. Incorporate SEO keywords naturally.\n"
                    "3. Sections/Subtitles are properly named in an engaging manner.\n"
                    "4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.\n"
                    "5. Proofread for grammatical errors and alignment with the brand's voice.\n"
                ),
                expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
                agent=writer,
            )

            edit = Task(
                description=("Proofread the given blog post for "
                             "grammatical errors and "
                             "alignment with the brand's voice."),
                expected_output="A well-written blog post in markdown format, "
                                "ready for publication, "
                                "each section should have 2 or 3 paragraphs.",
                agent=editor
            )
            crew = Crew(
                agents=[planner, writer, editor],
                tasks=[plan, write, edit],
            )
            result = crew.kickoff(inputs={"topic": topic})

            st.success("Blog successfully generated!")
            st.markdown("### 📝 Final Blog Output:")
            st.markdown(result)

