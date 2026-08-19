import os
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv

from state import chat_node, ModelState

load_dotenv()

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

    graph.add_edge(START, 'chat_node')
    graph.add_edge('chat_node', END)

    workflow = graph.compile(checkpointer=checkpointer, store=store)

if __name__ == '__main__':
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        Aemon.workflow.invoke({'query': input('User: ')}, config=config)