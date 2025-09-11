from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from langchain import hub
from typing_extensions import List, TypedDict
from langgraph.graph import START,StateGraph
from IPython.display import Image,display

urls = ['https://www.michaeljackson.com/news/michael-jacksons-rock-with-you-short-film-is-now-upscaled/']
loader =UnstructuredURLLoader(urls=urls)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=20)
all_splits =splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits,embeddding=HuggingFaceEmbeddings(),persis_directory="./my_chroma_db")

# to load it later
# vectorstore = Chroma(presist_directory="./my_chroma_db",embedding_function=HuggingFaceEmbeddings())

model_id = "tiiuae/falcon-7b"

text_generation_pipeline = pipeline("text-generation",model=model_id,model_kwargs={"torch_dtype":torch.bfloat6},max_new_tokens=200,device=0,temperature=0.7,top_k=50)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state:State):
    retrieved_docs = vectorstore.similarity_search(state["question"],k=1)
    return {"context":retrieved_docs}

def generate(state:State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question":state["question"],"context":docs_content})
    response = llm.invoke(messages)
    return {"answer":response}

graph_builder = StateGraph(State).add_sequence([retrieve,generate])
graph_builder.add_edge(START,"retrieve")
graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

response = graph.invoke({"question": "Michael jackson Rock with you 4K"})
print(response["answer"])