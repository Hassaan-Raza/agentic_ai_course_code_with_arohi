from langchain_community.chat_models import Ollama
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

ollama_model = Ollama(model="llama2")

def search_ddg(query: str):
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

def add(a: float, b: float) -> float:
    return a + b

def multiply(a: float, b: float) -> float:
    return a * b

math_agent = create_react_agent(
    model=ollama_model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Your job is to perform mathematical calculations. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=ollama_model,
    tools=[search_ddg],
    name="research_expert",
    prompt="You are a research expert. Your job is to perform web searches using DuckDuckGo. Do not perform any math."
)

workflow = create_supervisor(
    [research_agent, math_agent],
    model=ollama_model,
    prompt=(
        "You are a team supervisor. Your job is to manage a research expert and a math expert. "
        "For current events or tasks that require research, use research_agent. "
        "For math problems, use math_agent."
    )
)

app = workflow.compile()

result_1 = app.invoke({
    "messages": [{"role": "user", "content": "What is quantum computing?"}]
})
for m in result_1["messages"]:
    m.pretty_print()
