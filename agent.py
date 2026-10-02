import os
import uuid
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from state import ModelState, tools, tool_router
from create_agent import agent
from tools.permission_ui import ask_user_permission
import json

schema_path = "/home/adityagautam/Desktop/Projects/aemon-ai/tools/tool_schema.json"


load_dotenv()

tool_node = ToolNode(tools)
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

def text_of(msg) -> str:
    return msg.content if isinstance(msg.content, str) else "".join(
        b.get("text", "") for b in msg.content
        if isinstance(b, dict) and b.get("type") == "text")

def stamp(state: ModelState) -> dict:
    return {"llm_response": text_of(state["messages"][-1])}
    
class Aemon:
    graph = StateGraph(ModelState)

    graph.add_node('agent', agent)
    graph.add_node('stamp', stamp)
    graph.add_node('tool_node', tool_node)
    graph.add_node('tool1_router', tool_router)

    graph.add_edge(START, 'agent')
    graph.add_edge('agent', 'stamp')
    graph.add_edge('stamp', END)

    

async def main():
    async with AsyncConnectionPool(
    os.getenv('DATABASE_URL'),
    min_size=1,
    max_size=10,
    open=False,
    kwargs={"autocommit": True, "row_factory": dict_row}
)as pool:
        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        workflow = Aemon.graph.compile(checkpointer=checkpointer)
        while True:
            user_input = await asyncio.to_thread(input, 'User: ')
            if user_input.lower() in ['exit']:
                break

            initial_input = {"messages": [{"role": "user", "content": user_input}]}
            print("Jarvis: ", end="", flush=True)
            stream = workflow.astream_events(initial_input, config=config)

            async for event in workflow.astream_events(initial_input, config=config):
                if event["event"] == "on_chat_model_stream":
                    message_chunk = event["data"]["chunk"]

                    if message_chunk.content:
                        for block in message_chunk.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                print(block.get("text", ""), end="", flush=True)

            state = await workflow.aget_state(config=config)
            while state.tasks and any(t.interrupts for t in state.tasks):
                for task in state.tasks:
                    for intr in task.interrupts:
                        payload = intr.value
                        perm = await ask_user_permission(payload["command"])

                        async for event in workflow.astream_events(Command(resume=perm), config=config):
                            if event["event"] == "on_chat_model_stream":
                                message_chunk = event["data"]["chunk"]

                                if message_chunk.content:
                                    for block in message_chunk.content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            print(block.get("text", ""), end="", flush=True)
                state = await workflow.aget_state(config)

            print()
            #show = result["messages"][-1].content[1]['block']
            #print(f"Assistant: {show}")

if __name__ == "__main__":
    asyncio.run(main())