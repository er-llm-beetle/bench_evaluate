import os
import time

import httpx
import logging
import ollama
import pandas as pd
import Levenshtein

from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from dotenv import load_dotenv


load_dotenv()

# Configuration
BASE_URL_LLM = "https://integrate.api.nvidia.com/v1"
MODEL_LLAMA_3_1_405B = "meta/llama-3.1-405b-instruct"
MODEL_LLAMA_3_1_8B = "meta/llama-3.1-8b-instruct"
API_KEY_LLM = os.getenv('API_KEY_NVIDIA_LLM')
NUM_RETRIES = 3
BASE_SLEEP_TIME = 1


# SSL certificate problem fixing
httpx_client = httpx.Client(http2=True, verify=False)

# Initialize OpenAI client for NVIDIA
client_openai = OpenAI(base_url=BASE_URL_LLM, api_key=API_KEY_LLM, http_client=httpx_client)

# Initialize the Ollama client
client_ollama = ollama.Client()


