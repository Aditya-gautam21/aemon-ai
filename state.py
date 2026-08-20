from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END
from typing import TypedDict, Annotated
from llm import get_llm
from langchain_core.tools import tool
from tools.tool_call import run_command

class ModelState(TypedDict):
    messages: Annotated[BaseMessage, add_messages]
    query: str
    llm_response: str

tools = [run_command]

llm = get_llm()._llm.bind_tools(tools)

SYSTEM_PROMPT = [SystemMessage(content="You are a helpful assistant!")]

def chat_node(state: ModelState):
    user_msg = HumanMessage(content=state['query'])

    history = state.get('messages', [])
    prompt = SYSTEM_PROMPT + history + [user_msg]

    answer = llm.invoke(prompt)
    return {
        'messages': [user_msg, answer],
        'llm_response': answer.content
    }

def should_continue(state: ModelState):
    if state['messages'][-1].tool_calls:
        return 'tool_node'
    return END
     