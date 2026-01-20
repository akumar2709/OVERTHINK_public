import argparse
import os
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
from typing import List, Dict
import heapq
import math
from openai import OpenAI
from tqdm import tqdm
from openai import AzureOpenAI
import time
import anthropic
import requests
import json
from mistralai import Mistral
import os
from google import genai
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_ENV_LOADED = False


def _manual_load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = ".env"
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
    else:
        _manual_load_env(env_path)


def _get_env(name: str) -> str:
    _load_env()
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing {name} in .env or environment.")
    return value

def run_command_firework(prompt, model="deepseek_firework"):
    # print('prompt: ', prompt)
    client = OpenAI(
        api_key=_get_env("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    return response

def run_command_mistral(prompt, model="magistral-small-2509"):
    with Mistral(
        api_key=_get_env("MISTRAL_API_KEY")) as mistral:

        mist_res = mistral.chat.complete(model=model, messages=[
            {
                "content": prompt,
                "role": "user",
            },
        ], stream=False)
    return mist_res

def run_command_gemini(prompt):
    client = genai.Client(api_key=_get_env("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response    



def anthropic_run_command_sonnet_reasoning(user_prompt, system_prompt=None):
    client = anthropic.Anthropic(
            api_key=_get_env("ANTHROPIC_API_KEY"),
        )

    time_to_wait = 5
    for i in range(10):     
        try:            
            if system_prompt is None:
                response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=20000,
                    stream=False,        
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 16000
                    },      
                    messages=[{
                        "role": "user", 
                        "content": user_prompt
                    }])
            else:
                response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    system=system_prompt,
                    max_tokens=20000,
                    stream=False,        
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 16000
                    },       
                    messages=[{
                        "role": "user", 
                        "content": user_prompt
                    }])                
        except:
            print("here 3")
            time.sleep(time_to_wait)
            time_to_wait = time_to_wait*2
        else:
            break            
    print(response)        
    return {'text': response.content[1].text,
           'response':response}



def run_command(prompt, model):
    #if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=_get_env("OPENAI_API_KEY"))
    messages=[{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
                model=model,
               messages=messages
            )
    text = response.choices[0].message.content
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    cached_tokens = response.usage.prompt_tokens_details.cached_tokens
    return {'text': text, 'cached tokens': cached_tokens, 'reasoning tokens':reasoning_tokens, "entire respose":response}
