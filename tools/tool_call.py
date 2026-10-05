import subprocess
from langchain_core.tools import tool
from langgraph.config import get_store
import uuid
import hashlib
import re

_ws = re.compile(r"\s+")

def _make_key(fact: str):
    normalised = _ws.sub(" ", fact.strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]

@tool
def run_command(command: str) -> str:
    """Run a shell command."""

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

@tool
def remember(fact: str, category:str = "other", importance: float = 0.3) -> str:
    """Save a durable fact to long-term memory. Use when the user states a
    preference, environment detail, or asks you to remember something."""

    store = get_store()
    store.put(("memories", "user1"), _make_key(fact), {
                "fact": fact, "category": category, "importance": importance
            })
    return 'saved'

@tool
def recall(query: str, limit: int = 5):
    """Search long-term memory for facts relevant to query."""

    store = get_store()
    hits = store.search(("memories", "user1"), query=query, limit=limit)
    return hits if hits else "No memory found."
