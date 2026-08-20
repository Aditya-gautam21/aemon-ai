import os
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv
import json

from state import chat_node, ModelState, should_continue, tools
from tools.tool_call import run_command

load_dotenv()

tool_node = ToolNode(tools)

pool = ConnectionPool(
    os.getenv('DATABASE_URL'),
    min_size=1,
    max_size=10,
    kwargs={"autocommit": True, "row_factory": dict_row}
)
store = PostgresStore(pool)
store.setup()

checkpointer = PostgresSaver(pool)
checkpointer.setup()
        
class Aemon:
    graph = StateGraph(ModelState)

    graph.add_node('chat_node', chat_node)
    graph.add_node('tool_node', tool_node)

    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', should_continue, {
        'tool_node': 'tool_node',
        END: END
    })
    graph.add_edge('tool_node', 'chat_node')

    workflow = graph.compile(checkpointer=checkpointer, store=store)

if __name__ == '__main__':
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        user_input = input('User: ')
        result = Aemon.workflow.invoke({'query': user_input}, config=config)
        print(f"Assistant: {result['llm_response']}")