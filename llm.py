import subprocess
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import time 

class LocalLLM:
    def __init__(self):
        cmd = [
            "llama-server",
            "-m", "/home/adityagautam/llama.cpp/models/google_gemma-4-E4B-it-Q4_K_M.gguf",
            "--host", "0.0.0.0",
            "--port", "8080",
            "-ngl", "999",
            "-c", "8192",
            "--flash-attn", "on",
            "--jinja"
        ]

        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            )
        
        time.sleep(10)

        self._llm = ChatAnthropic(
            base_url="http://0.0.0.0:8080",
            api_key="not-needed",
            model="Qwen3.5-4B-Q4_K_M",
            streaming=True
        )

class RemoteLLM:
    def __init__(self):
        self._llm = ChatOpenAI(model="openai/gpt-oss-120b:free", base_url="https://openrouter.ai", api_key=os.getenv('OPENROUTER_API_KEY'), temperature=0.5, streaming=True)

    def invoke(self, messages) -> str:
        return self._llm.invoke(messages).content

    def bind_tools(self, tools):
            return self._llm.bind_tools(tools)

def get_llm(local: bool = True):
    return (LocalLLM() if local else RemoteLLM())
