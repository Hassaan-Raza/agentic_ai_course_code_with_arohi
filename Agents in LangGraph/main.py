from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama.llms import OllamaLLM
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=700)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wiki_tool]
llm = OllamaLLM(model="llama2")



agent_executor = create_react_agent(llm,tools)
response = agent_executor.invoke({"messages":[HumanMessage(content="hi!")]})

print(response["messages"])
