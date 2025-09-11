import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load .env file
load_dotenv()


async def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp"  # URL of your MCP server
            }
        }
    )

    tools = await client.get_tools()

    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]

        # Check if the message has a 'tool_calls' attribute (i.e., it's a tool response)
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END

    async def call_model(state: MessagesState):
        messages = state["messages"]
        prompt = " ".join([message.content for message in messages])

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(inputs['input_ids'], max_length=150, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        tool_name = detect_tool_from_response(response)

        if tool_name:
            tool_response = await tools[0].invoke({"query": prompt})
            response = f"Tool ({tool_name}) Response: {tool_response['content']}"

        return {"messages": [response]}

    def detect_tool_from_response(response: str):
        if "calculate" in response:
            return "math"
        else:
            return None

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)

    graph = builder.compile()

    result = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
