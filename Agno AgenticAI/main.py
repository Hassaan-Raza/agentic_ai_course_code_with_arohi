from agno.agent import Agent
from agno.models.ollama import Ollama

agno_agent=Agent(
    model = Ollama(id="llama3.2"),
    description = "Agno Q&A agent",
    markdown=False
)

agno_agent.print_response("Do you know what is Lego Brick Life, which was made by epic games?its a game",stream=True)