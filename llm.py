import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

SERVER_LOG = Path(__file__).resolve().parent / ".llama-server.log"

# Hosts that mean "every interface" -- clients must connect to loopback instead.
WILDCARD_HOSTS = ("0.0.0.0", "::")


class LocalLLM:
    """A llama.cpp ``llama-server`` process serving one local GGUF model.

    Configuration comes entirely from ``.env`` (written by ``python setup.py``):

        LLAMA_SERVER_BIN   path to the llama-server executable
        LOCAL_MODEL_PATH   the .gguf file to serve
        LOCAL_MODEL_NAME   model id advertised to the client
        LLAMA_HOST         bind address          (default 0.0.0.0)
        LLAMA_PORT         port                  (default 8080)
        LLAMA_CTX          context length        (default 8192)
        LLAMA_NGL          GPU layers, 999 = all (default 999)
        LLAMA_EXTRA_ARGS   extra flags           (default --flash-attn on --jinja)
        LLAMA_BOOT_TIMEOUT seconds to wait for the model to load (default 180)

    If a server is already answering on that port it is reused rather than
    started twice, so ``agent.py`` and a manual ``llama-server`` can coexist.
    """

    def __init__(self):
        self._binary = os.getenv("LLAMA_SERVER_BIN", "llama-server")
        self._model = os.getenv("LOCAL_MODEL_PATH", "")
        self._host = os.getenv("LLAMA_HOST", "0.0.0.0")
        self._port = int(os.getenv("LLAMA_PORT", "8080"))
        self._proc: subprocess.Popen | None = None

        self._start()

        self._llm = ChatAnthropic(
            base_url=f"http://{self.connect_host}:{self._port}",
            api_key="not-needed",
            model=os.getenv("LOCAL_MODEL_NAME") or Path(self._model).stem or "local-model",
            streaming=True
        )

    @property
    def connect_host(self) -> str:
        return "127.0.0.1" if self._host in WILDCARD_HOSTS else self._host

    def command(self) -> list[str]:
        cmd = [
            self._binary,
            "-m", self._model,
            "--host", self._host,
            "--port", str(self._port),
            "-ngl", os.getenv("LLAMA_NGL", "999"),
            "-c", os.getenv("LLAMA_CTX", "8192"),
        ]
        return cmd + os.getenv("LLAMA_EXTRA_ARGS", "--flash-attn on --jinja").split()

    def _start(self) -> None:
        if self.is_healthy():
            return
        if not self._model:
            raise RuntimeError("LOCAL_MODEL_PATH is not set -- run `python setup.py` first.")
        binary = shutil_which(self._binary)
        if not binary:
            raise RuntimeError(f"llama-server not found at {self._binary!r} -- "
                               "run `python setup.py` to install it.")
        self._binary = binary

        # stdout/stderr go to a log file: piping them would fill the OS buffer
        # and block the server once it has written ~64KB of debug output.
        log = open(SERVER_LOG, "ab")
        self._proc = subprocess.Popen(
            self.command(),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(Path(self._binary).parent),
        )

        deadline = time.time() + float(os.getenv("LLAMA_BOOT_TIMEOUT", "180"))
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode}. "
                    f"See {SERVER_LOG.name} for the reason.")
            if self.is_healthy():
                return
            time.sleep(1.0)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy within the boot timeout "
                           f"(model {Path(self._model).name}, {self._port}). See {SERVER_LOG.name}.")

    def is_healthy(self) -> bool:
        """True when a server on the configured port reports status ok."""
        try:
            with urllib.request.urlopen(f"http://{self.connect_host}:{self._port}/health",
                                         timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def stop(self) -> None:
        """Terminate only the process this instance started."""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def invoke(self, messages) -> str:
        return self._llm.invoke(messages).content

    def bind_tools(self, tools):
        return self._llm.bind_tools(tools)


class RemoteLLM:
    def __init__(self):
        self._llm = ChatOpenAI(model="openai/gpt-oss-120b:free", base_url="https://openrouter.ai", api_key=os.getenv('OPENROUTER_API_KEY'), temperature=0.5, streaming=True)

    def invoke(self, messages) -> str:
        return self._llm.invoke(messages).content

    def bind_tools(self, tools):
            return self._llm.bind_tools(tools)


def shutil_which(binary: str) -> str | None:
    """Absolute path to the binary, accepting bare names and relative paths."""
    path = Path(binary).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(binary)


_instances: dict[bool, LocalLLM | RemoteLLM] = {}


def get_llm(local: bool = True):
    """Return the shared LLM wrapper, starting llama-server on first use."""
    if local not in _instances:
        _instances[local] = LocalLLM() if local else RemoteLLM()
    return _instances[local]
