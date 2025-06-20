from crewai import Task
from utils.tools import yt_tool
from utils.agents import blog_researcher, blog_writer

#Research Task
research_task=Task(
    description=(
        'Identify the video {topic}.'
        'Get detailed information about the video from the channel video'
    ),
    expected_output='A comprehensive 3 paragraphs long report based on the {topic} of video content.',
    tools=[yt_tool],
    agent=blog_researcher
)

#Write task
write_task=Task(
    description=(
        'Get the info from the youtube channel channel on the {topic}'
    ),
    expected_output='Summarize the info from the youtube channel video on the topic {topic} and create the content for the blog',
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_file='new-blog-post.md'
)