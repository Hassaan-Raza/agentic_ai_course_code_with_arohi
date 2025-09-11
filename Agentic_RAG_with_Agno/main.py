from agno.models.ollama import Ollama
from agno.agent import Agent
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.vectordb.pgvector import PgVector

knowledge_base = ArxivKnowledgeBase(
    queries=["Generative AI", "Machine Learning"],
    # Table name: ai.arxiv_documents
    vector_db=PgVector(
        table_name="arxiv_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
    ),
)

agent = Agent(
    model=Ollama(id="llama3.2"),
    knowledge=knowledge_base,
    search_knowledge=True
)

agent.knowledge.load(recreate=False)

user_prompt = input("Ask a question from the knowledge base: ")

agent.print_response(user_prompt,user_id="user_1",stream=True)