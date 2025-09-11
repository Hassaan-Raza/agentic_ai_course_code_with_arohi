from crewai_tools import WebsiteSearchTool
from langchain_community.chat_models import ChatOllama
from crewai import Agent,Task,Crew
import os
model_name = "llama2"
llm = ChatOllama(model=model_name,temperature=0)
web_search_tool = WebsiteSearchTool()

researcher = Agent(role='Market Research Analyst',
                   goal='Provide up-to-date market analysis of the Pop music Industry',
                   backstory= 'An expert analysis with a keen eye for Pop music industry trends',
                   tools = [web_search_tool],
                   verbost=True,
                   llm=llm)
writer = Agent(role='Content Writer',
                   goal='Craft engaging blog posts about the Pop music Industry',
                   backstory= 'A skilled writer with a passion for Pop music trends',
                   tools = [],
                   verbost=True,
                   llm=llm)
research=Task(description= 'Search the web for the latest trends in the Pop music industry and provide a summarized report.',
              expected_output='A concise summary of the top 3 trending developments in Pop music, with brief insights into their cultural or commercial impact.',
              agent=researcher)
write = Task(description = 'Write an engaging blog post about current Pop music industry trends based on the research analyst’s summary.',
             expected_output='A well-structured, 4-paragraph blog post in markdown format that’s easy to read and captures the excitement of the current Pop music scene.',
             agent=writer,
             output_file='blog_post/pop_music_trends.txt')

crew = Crew(agents=[researcher,writer],
            tasks=[research,write],
            verbose=True,
            planning=True)

crew.kickoff()