from langchain.agents import create_agent
from tools.tool_call import run_command, remember, recall
from llm import get_llm
from tools.permission import PermissionMiddleware
from dotenv import load_dotenv
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware, TodoListMiddleware
from langgraph.store.base import BaseStore

from prompts import JUDGE_PROMPT, aemon_personality
from state import ModelState

load_dotenv()

model = get_llm(local=True)._llm
schema_path = "/home/adityagautam/Desktop/Projects/aemon-ai/tools/tool_schema.json"

chat_agent = create_agent(
    model=model,
    tools=[run_command, remember, recall],
    middleware=[PermissionMiddleware(schema_path=schema_path),
                TodoListMiddleware(),
                ClearToolUsesEdit_and_friends := ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=6000, keep=2)]),
                ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
                ToolRetryMiddleware(max_retries=7, on_failure="return_message")],
    system_prompt=aemon_personality
)

memory_agent = create_agent(
    model=model,
    system_prompt=JUDGE_PROMPT
)

async def memory_gate(state: ModelState, *, store: BaseStore):
    history = state['messages'][-2]
    verdict = memory_agent.ainvoke(history)

    if verdict['store']:
        await store.aput(("memories", "user1"), {
            "fact": verdict['fact'], "category": verdict['category'], "importance": verdict['importance']
        })
    return {}
