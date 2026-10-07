import re
from enum import Enum
from pathlib import Path
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

# Tool args whose name hints they carry a filesystem path or a shell command.
_PATH_ARG = re.compile(r"(path|file|dir|folder)", re.IGNORECASE)
_COMMAND_ARG = re.compile(r"(command|cmd|shell)", re.IGNORECASE)

COMMAND_ALLOW = ["ls", "pwd", "git status", "git diff",
                 "cat", "head", "tail", "find", "grep"]
COMMAND_DENY = ["sudo", "shutdown", "reboot", "mkfs", "dd", "chmod", "chown"]


def build_tool_schema(tools) -> dict:
    """Derive the permission schema from the tools themselves.

    Each tool contributes an entry keyed by its name: args that look like
    shell commands get the command policy, args that look like filesystem
    paths get a path policy rooted at the current working directory,
    everything else is allowed to run ungated.
    """
    root = str(Path.cwd())
    schema = {}
    for t in tools:
        name = getattr(t, "name", None) or type(t).__name__
        try:
            props = t.args_schema.schema()["properties"]
        except Exception:
            props = {}

        if any(_COMMAND_ARG.search(arg) for arg in props):
            schema[name] = {"enabled": True, "permissions": {
                "default": "ask",
                "allow": COMMAND_ALLOW,
                "deny": COMMAND_DENY,
            }}
        elif any(_PATH_ARG.search(arg) for arg in props):
            schema[name] = {"enabled": True, "permissions": {
                "default": "ask",
                "allow_paths": [f"{root}/**"],
                "deny_paths": [str(Path.home() / ".ssh/**"), str(Path.home() / ".aws/**")],
            }}
        else:
            schema[name] = {"enabled": True, "permissions": {"default": "allow"}}
    return schema

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
    def __init__(self, tools):
        super().__init__()
        # Schema is generated from the tool definitions, not a hand-maintained file.
        self.config = build_tool_schema(tools)

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
