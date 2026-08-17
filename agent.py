from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import BaseMessage, add_messages
from typing import TypedDict, Annotated
from llm import get_llm
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm, local = get_llm(local=True)
class ModelState(TypedDict):
    messages: Annotated[BaseMessage, add_messages]
    initial_state: list[dict]
    query: str
    llm_response: str

store = PostgresStore.from_conn_string(conn_string=os.getenv('DATABASE_URL'))
store.setup()

checkpointer = PostgresSaver.from_conn_string(conn=os.getenv('DATABASE_URL'), serde=SerializerProtocol())
checkpointer.setup()

def initial_query(state: ModelState):
        global llm
        state['initial_state'] = [
            {'role': 'system', 'content': 'You are an helpful assistant!'},
            {'role': 'assistant', 'content': 'Hi! How can i help you today?'}
        ]

        state['query'] = input('User: ')

        state['initial_state'].append({'role': 'user', 'content': state['query']})

        if local:
            response = llm.create_chat_completion(
                messages=state['initial_state'],
                stream=False
            )

            state['llm_response'] = response['choices'][0]['message']['content']
            return state
        else:
             response = llm.invoke(input=state['initial_state'], stream=False)

             state['llm_response'] = response.content
             return state
        
class Aemon:
    graph = StateGraph(ModelState)

    graph.add_node('initial_query', initial_query)

    graph.add_edge(START, 'initial_query')
    graph.add_edge('initial_query', END)

    workflow = graph.compile(checkpointer=checkpointer, store=store)

if __name__ == '__main__':
    initial_payload: ModelState = {
        "initial_state": [],
        "query": "",
        "llm_response": ""
    }
    
    # Run the graph and store the resulting final state
    final_state = Aemon.workflow.invoke(initial_payload)
    print(final_state['llm_response'])