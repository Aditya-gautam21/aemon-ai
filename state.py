from langgraph.graph import StateGraph
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated
from llm import get_llm

class ModelState(TypedDict):
    messages: Annotated[BaseMessage, add_messages]
    query: str
    llm_response: str

llm, local = get_llm()

SYSTEM_PROMPT = [SystemMessage(content="You are a helpful assistant!")]

def chat_node(state: ModelState):
    user_msg = HumanMessage(content=state['query'])

    history = state['messages']
    prompt = SYSTEM_PROMPT + history + user_msg

    if local:
        response = llm.create_chat_completion(messages=to_openai())
