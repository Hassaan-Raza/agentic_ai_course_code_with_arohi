import asyncio
import os
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

async def main():
    owm_key = os.getenv("OWM_API_KEY")

    if not owm_key:
        raise ValueError("OWM KEY NOT FOUND")

    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "stdio",
                "command": "C:/Users/Moonwalking/PycharmProjects/MCP_server/mcp-openweather/mcp-weather.exe",
                "args": [],
                "env": {"OWM_API_KEY": owm_key}
            }
        }
    )
    tools = await client.get_tools()

    # GPT-2 Model and Tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad token to be eos_token
    tokenizer.pad_token = tokenizer.eos_token  # Setting pad_token to eos_token

    def call_model(state: MessagesState):
        prompt = state["messages"]

        # If the prompt is a list of HumanMessage objects, extract their content
        if isinstance(prompt, list):
            # Extracting the 'content' field from each HumanMessage
            prompt = " ".join([message.content if hasattr(message, 'content') else str(message) for message in prompt])
        elif isinstance(prompt, str):
            pass  # It's already a string, no need to change.
        else:
            raise ValueError("Expected input type: str or list of HumanMessage objects")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

        # Generate response
        outputs = model.generate(
            inputs['input_ids'],
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,  # Use eos_token_id to stop the generation
            eos_token_id=tokenizer.eos_token_id,  # Stop generation at eos token
            no_repeat_ngram_size=2,  # Prevent repetition of phrases
            attention_mask=inputs.get("attention_mask")  # Make sure attention_mask is used
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()

    print("\n---Weather Query---")

    result = await graph.ainvoke({"messages": "Whats the weather in Texas Today?"})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
