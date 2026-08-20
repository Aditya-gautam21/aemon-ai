import os
from dotenv import load_dotenv
from llama_cpp import Llama
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import convert_to_openai_messages
from langchain_community.chat_models import ChatLlamaCpp

load_dotenv()

class LocalLLM:
    def __init__(self):
        self._llm = ChatLlamaCpp(model_path=os.getenv('LOCAL_MODEL_PATH'), n_gpu_layers=24, n_batch=512, n_ctx=8192, temperature=0.5, verbose=False)

    def invoke(self, messages) -> str:
        return self._llm.invoke(messages).content

    def bind_tools(self, tools):
        return self._llm.bind_tools(tools)

class RemoteLLM:
    def __init__(self):
        self._llm = ChatDeepSeek(api_key=os.getenv('DEEPSEEK_API_KEY'), temperature=0.5)

    def invoke(self, messages) -> str:
        return self._llm.invoke(messages).content

    def bind_tools(self, tools):
            return self._llm.bind_tools(tools)

def get_llm(local: bool = True):
    return (LocalLLM() if local else RemoteLLM())
