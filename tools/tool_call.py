import subprocess
from langchain_core.tools import tool
import json

with open("/home/adityagautam/Desktop/Projects/aemon-ai/tools/tool_schema.json", "r") as t:
    tool_schema = json.load(t)

@tool(description="Run a shell command.")
def run_command(command: str) -> str:
    if command in tool_schema['permissions']['allow']:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        return (
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return "Command not allowed!"

"""@tool
def run_sudo_command(command: str) -> str:
    result = subprocess.run(
            command,
            shell=True,
            text=True,
            timeout=30
        )
    
    return (
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

@tool
def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        f.read()"""