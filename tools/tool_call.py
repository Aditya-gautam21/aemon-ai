import os
import subprocess
from langchain_core.tools import tool
from langgraph.config import get_store
import uuid
import hashlib
import re

_ws = re.compile(r"\s+")

def _clean_env() -> dict:
    """Return a copy of os.environ with conda's library paths stripped.

    The agent runs inside a conda env whose libreadline.so.8 lacks symbols the
    system shell links against (e.g. rl_print_keybinding), which makes every
    subprocess shell fail at startup with exit code 127.
    """
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)

    conda_lib = os.path.join(env.get("CONDA_PREFIX", ""), "lib") if env.get("CONDA_PREFIX") else ""
    lib_path = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in lib_path.split(":") if p and p != conda_lib and "mambaforge" not in p and "miniconda" not in p and "anaconda" not in p]
    if parts:
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env

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
        timeout=30,
        env=_clean_env()
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
