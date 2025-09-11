from agno.agent import Agent
from agno.models.ollama import Ollama
from datetime import datetime
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.storage.sqlite import SqliteStorage
from rich.pretty import pprint

db_file = "tmp/agent_storage.db"
user_id = "Moonwalking"

memory = Memory(db=SqliteMemoryDb(table_name="user_memories",db_file=db_file))

agent = Agent(
    model=Ollama(id="llama3.2"),
    description="You are an intelligent AI with memory",
    memory=memory,
    storage=SqliteStorage(table_name="agent_session",db_file=db_file),
    enable_user_memories=True,
    add_history_to_messages=True,
    num_history_runs=3,
    session_id="my_chat_session",
    markdown=True
)

agent.print_response("I Love Michael Jackson and his music. Who is your favorite singer?",user_id=user_id,stream=True)

print("Current memories about user: ")
user_memories = memory.get_user_memories(user_id=user_id)
pprint([{"memory":m.memory,"topics":m.topics}for m in user_memories])

agent.print_response("Which singer do i love? i also love rainbow six siege x",user_id=user_id)