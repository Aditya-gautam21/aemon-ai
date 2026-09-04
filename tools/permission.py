import json
from enum import Enum
import questionary
from questionary import Choice
from typing import Any, Awaitable, Callable
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolCallRequest
from langgraph.types import Command

from tools.permission_ui import ask_user_permission

class Permission(Enum):
    ALLOW = "allow"
    SESSION = "allow_for_this_session"
    ALWAYS = "always_allow"
    DENY = "deny"



class PermissionMiddleware(AgentMiddleware):
    def __init__(self, schema_path):
        super().__init__()
        with open (schema_path) as f:
            self.config = json.load(f)['tools']

    async def permission_ui(self, command):
        answer = await questionary.select(
            message=f"{command}",
            choices=[
                Choice(title="Allow once", value=Permission.ALLOW),
                Choice(title="Allow for this session", value=Permission.SESSION),
                Choice(title="Always allow", value=Permission.ALWAYS),
                Choice(title="Deny", value=Permission.DENY)
            ],
        ).ask()

        return answer

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
                tool_call_id = request.tool_call_id
            )

    async def check(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]) -> Permission:
        tool_name, args = request.name, request.args

        tool_config = self.config.get(tool_name, {})
        perms = tool_config.get('permissions', {})

        if tool_name == 'run_command':
            cmd = args.get('command', '')
            call = self._check_command(cmd, perms)

            return await self.tool_next(call, request, handler)

    async def _check_command(self, cmd, perms):
        for allowed in (perms.get("allow", []) or perms.get("allow_for_this_session", [])):
            if cmd == allowed:
                yield Permission.ALLOW
            else:
                return await self.permission_ui()


