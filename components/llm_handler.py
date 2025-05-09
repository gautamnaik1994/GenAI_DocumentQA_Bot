from langchain_community.llms import LlamaCpp
from langchain_together import ChatTogether
import os


# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
LOCAL_MODEL_PATH = "/Users/gautamnaik/models/Mistral-7B-Instruct-v0.3.Q8_0.gguf"


def get_cloud_llm():
    return ChatTogether(
        api_key=os.getenv("TOGETHER_AI_API_KEY"),
        temperature=0.0,
        model=MODEL_NAME
    )


def get_local_llm():
    llm = LlamaCpp(
        model_path=LOCAL_MODEL_PATH,
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=32,
        temperature=0.7
    )
    return llm


def get_llm(use_cloud: bool):
    if use_cloud:
        return get_cloud_llm()
    else:
        return get_local_llm()
