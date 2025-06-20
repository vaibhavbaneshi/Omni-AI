import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crewai import Crew, Process
from utils.agents import blog_researcher, blog_writer
from utils.task import research_task, write_task
from utils.llm import llm
from common.streamlit_imports import st

#Forming the tech-focused crew with some enhanced configurations

def run_blog_writer():
    crew=Crew(
        agents=[blog_researcher, blog_writer],
        tasks={research_task, write_task},
        process=Process.sequential,
        memory=False,
        cache=False,
        max_rpm=100,
        share_crew=True,
        llm=llm,
        embedder={     # 👇 This disables embedding usage completely
            "provider": "none"
        }
    )

    ##start the tash execution process the enhanced feedback
    result=crew.kickoff(inputs={'topic':'Why Trump’s move to sell AMRAAM to Turkey alarms India? Ankit Agrawal Study IQ'})

    st.write(result)