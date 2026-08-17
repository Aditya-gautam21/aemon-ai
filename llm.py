import os
from dotenv import load_dotenv
from llama_cpp import Llama
from langchain_deepseek import ChatDeepSeek

load_dotenv()
_llm = None

def get_llm(local: bool = True):
    global _llm
    if _llm is None:
        if local:
            _llm = Llama(
                model_path=os.getenv('LOCAL_MODEL_PATH'),
                n_gpu_layers=28,
                temperature=0.5
            )
        else:
            _llm = ChatDeepSeek(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                temperature=0.5
            )

    return _llm, local