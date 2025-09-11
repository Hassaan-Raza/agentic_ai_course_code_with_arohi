from IPython.display import Image, display
from langgraph.graph import StateGraph, START
from langchain_ollama.llms import OllamaLLM
import requests
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState

from langgraph.prebuilt import ToolNode, tools_condition

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = OllamaLLM(model="llama2")


def search_duckduckgo(query: str):
    """
        Searches DuckDuckGo using LangChain's DuckDuckGoSearchRun tool.

        Args:
            query (str): The search query to perform.

        Returns:
            str: The search result content.
        """
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

tools = [search_duckduckgo, add, multiply]

def llm_with_tools(messages: list):
    query = None
    result = None

    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content

            if "search:" in content:
                query = content.replace("search:", "").strip()
                result = search_duckduckgo(query)
            elif "multiply" in content:
                nums = [int(s) for s in content.split() if s.isdigit()]
                if len(nums) == 2:
                    result = multiply(nums[0], nums[1])
            elif "add" in content:
                nums = [int(s) for s in content.split() if s.isdigit()]
                if len(nums) == 2:
                    result = add(nums[0], nums[1])

    if result is None:
        result = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": result}]}


def chatbot(state: State):
    return llm_with_tools(state["messages"])

from langgraph.prebuilt import ToolNode, tools_condition

graph_builder = StateGraph(State)

graph_builder.add_node("assistant", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "assistant")
graph_builder.add_conditional_edges("assistant", tools_condition)
graph_builder.add_edge("tools", "assistant")

react_graph = graph_builder.compile()

display(Image(react_graph.get_graph().draw_mermaid_png()))

response = react_graph.invoke(
    {"messages": [HumanMessage(content="search: download fortnite")]})

for m in response['messages']:
    m.pretty_print()
