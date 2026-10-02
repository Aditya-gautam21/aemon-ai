import json
from enum import Enum
import questionary
from questionary import Choice
from typing import Any, Awaitable, Callable
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt

class Permission(Enum):
    ALLOW = "allow"
    SESSION = "allow_for_this_session"
    ALWAYS = "always_allow"
    DENY = "deny"

async def permission_ui(command):
        answer = questionary.select(
            message=f"{command}",
            choices=[
                Choice(title="Allow once", value=Permission.ALLOW),
                Choice(title="Allow for this session", value=Permission.SESSION),
                Choice(title="Always allow", value=Permission.ALWAYS),
                Choice(title="Deny", value=Permission.DENY)
            ],
        ).ask_async()

        return answer

class PermissionMiddleware(AgentMiddleware):
    def __init__(self, schema_path):
        super().__init__()
        with open (schema_path) as f:
            self.config = json.load(f)['tools']

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]) -> Permission:
        tool_name, args = request.tool_call["name"], request.tool_call["args"]
        tool_config = self.config.get(tool_name, {})
        perms = tool_config.get('permissions', {})

        if tool_name == 'run_command':
            cmd = args.get('command', '')
            decision = await self._check_command(cmd, perms)
        else:
            return await handler(request)

        return await self.tool_next(decision, request, handler)

    async def tool_next(self, call, request, handler):
            if call == Permission.ALLOW:
                return await handler(request)
            if call == Permission.SESSION:
                return await handler(request)
            if call == Permission.ALWAYS:
                return await handler(request)
            if call == Permission.DENY:
                return ToolMessage(
                    content="Command denied by the user!",
                    tool_call_id = request.tool_call["id"]
                )
            
    async def _check_command(self, cmd, perms):
        allow_list = perms.get("allow", []) + perms.get("allow_for_this_session", [])
        for allowed in allow_list:
            if cmd == allowed:
                return Permission.ALLOW

        answer = interrupt({
            "type": "permission_request",
            "command": cmd
        })

        return Permission(answer)
