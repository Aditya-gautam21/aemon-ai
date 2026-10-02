from langchain.agents import create_agent
from tools.tool_call import run_command
from llm import get_llm
from tools.permission import PermissionMiddleware
from dotenv import load_dotenv
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware

load_dotenv()

model = get_llm(local=True)._llm
schema_path = "/home/adityagautam/Desktop/Projects/aemon-ai/tools/tool_schema.json"

agent = create_agent(
    model=model,
    tools=[run_command],
    middleware=[PermissionMiddleware(schema_path=schema_path),
                ClearToolUsesEdit_and_friends := ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=6000, keep=2)]),
                ModelCallLimitMiddleware(thread_limit=12, exit_behavior="end"),
                ToolRetryMiddleware(max_retries=1, on_failure="return_message")],
    system_prompt="You are a helpful assistant!"
)
