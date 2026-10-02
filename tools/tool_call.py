import subprocess
from langchain_core.tools import tool

@tool
def run_command(command: str) -> str:
    """Run a shell command."""
    #_check_permission('run_command', {'command': command})

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )

    print(result)

    return (
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@tool
def read_file(filepath: str) -> str:
    """Read a file and return its contents."""
    _check_permission('read_file', {'filepath': filepath})

    with open(filepath, 'r') as f:
        return f.read()


@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file."""
    _check_permission('write_file', {'filepath': filepath, 'content': content})

    with open(filepath, 'w') as f:
        f.write(content)
    return f"Written {len(content)} bytes to {filepath}"