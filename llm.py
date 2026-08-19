import os
from dotenv import load_dotenv
from llama_cpp import Llama
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import convert_to_openai_messages

load_dotenv()

class LocalLLM:
    def __init__(self):
        self._llm = Llama(model_path=os.getenv('LOCAL_MODEL_PATH'), n_gpu_layers=28, temperature=0.5)

    def invoke(self, messages) -> str:
        r = self._llm.create_chat_completion(messages=convert_to_openai_messages(messages), stream=False)
        return r['choices'][0]['message']['content']

class RemoteLLM:
    def __init__(self):
        self._llm = ChatDeepSeek(api_key=os.getenv('DEEPSEEK_API_KEY'), temperature=0.5)

    def invoke(self, messages) -> str:
        return self._llm.invoke(messages).content

def get_llm(local: bool = True):
    return (LocalLLM() if local else RemoteLLM(), local)
