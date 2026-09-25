import os

from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel, ModelSettings
from agents.memory import SessionInputCallback

import lmstudio as lms
import yaml
SERVER_API_HOST = "localhost:1234"
lms.configure_default_client(SERVER_API_HOST)

# from .weave_utils import LLM_RATE_LIMITS

def setup_api_keys():
    _api_keys_path = Path(__file__).resolve().parent.parent / "api_keys.yaml"
    if not _api_keys_path.exists():
        _api_keys_path = Path("api_keys.yaml")

    if _api_keys_path.exists():
        with open(_api_keys_path, "r", encoding="utf-8") as _f:
            _api_keys = yaml.safe_load(_f) or {}
        for _k, _v in _api_keys.items():
            if _v is not None:
                os.environ[str(_k)] = str(_v)

# Load API keys into environment variables
setup_api_keys()

github_token = os.environ.get("GITHUB_TOKEN", "")
openai_token = os.environ.get("OPENAI_API_KEY", "")
gemini_token = os.environ.get("GEMINI_API_KEY", None)
claude_token = os.environ.get("CLAUDE_API_KEY", None)

github_url = "https://models.github.ai/inference"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
claude_url = "https://api.anthropic.com/v1/"


def load_lmstudio_model(model_name: str, single_gpu: bool = True):
    client = lms.get_default_client()
    if 'gemma-4-31b' in model_name.lower():
        if single_gpu:
            context_length = 42000
        else:
            context_length = 260000
    elif 'gemma-4-e4b' in model_name.lower():
        context_length = 131072
    else:
        context_length = 262144

    model =  client.llm.model(
        model_name,
        config={
            "contextLength": context_length,
            "gpu": {"main_gpu": 0, "disabled_gpus": [1]} if single_gpu else {}
        },
    )


def create_openai_model(model_name: str, single_gpu: bool = True): 

    if "openai" in model_name.lower(): 
        api_key = github_token
        endpoint = github_url
        model = OpenAIChatCompletionsModel
    elif "gemini" in model_name.lower(): 
        api_key = gemini_token
        endpoint = gemini_url
        model = OpenAIChatCompletionsModel
    elif "claude" in model_name.lower():
        api_key = claude_token
        endpoint = claude_url
        model = OpenAIChatCompletionsModel
    elif 'gemma' in model_name.lower() or 'qwen' in model_name.lower() or 'oss' in model_name.lower():
        model = OpenAIResponsesModel
        endpoint = "http://localhost:1234/v1"
        api_key = "lm-studio"
        load_lmstudio_model(model_name, single_gpu)
    else:
        api_key = openai_token
        endpoint = None
        model = OpenAIResponsesModel

    client = AsyncOpenAI(
        base_url=endpoint,
        api_key=api_key,
    )

    openai_model = model( 
            model=model_name,
            openai_client=client,
        )
    
    return openai_model


def session_input_callback(history_items, new_items, **kwargs):
    # Simple example: combine history and new items
    for it in new_items:
        if isinstance(it,list):
            pass
        else:
            history_items.append(it)
            
    return history_items