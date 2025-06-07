import torch
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline, 
)
import os
from huggingface_hub import snapshot_download

# Set environment variables for offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

hf_models = [
    # Free, high quality
    "mistralai/Mixtral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",

    # Need license
    "meta-llama/Llama-2-7b-chat",
    "meta-llama/Llama-2-13b-chat",
    "meta-llama/Llama-2-7b0-chat",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",

    # Good for inference, work well with transformer of Hugging Face
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",

    # Optimized for dialogual
    "HuggingFaceH4/zephyr-7b-beta",

    # Good ranking
    "teknium/OpenHermes-2.5-Mistral-7B",

    # For dialogual and reasoning
    "NousResearch/Nous-Hermes-2-Mistral-7B",
    "NousResearch/Nous-Capybara-7B",

    # Good for edge and small device
    "microsoft/phi-2",

    # Light model, good for local
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/gemma-7b-it",
    "Qwen/Qwen1.5-7B-Chat",
    "01-ai/Yi-1.5-9B-Chat"
]

other_models = [
    "gemini-2.0-flash",
]

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

def download_model_if_not_exists(model_name: str) -> None:
    """
    Download model if it doesn't exist locally
    """

    try:
        # Check if model exists in cache
        model_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", model_name.replace("/", "--"))
        if not os.path.exists(model_path):
            print(f"Downloading model {model_name}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
    except Exception as e:
        print(f"Warning: Could not download model {model_name}: {str(e)}")
        print("Please ensure you have internet connection or the model is already downloaded.")


def load_llm(model_name: str) -> Any:
    '''
        Load LLm model
    '''

    if model_name == "gemini-2.0-flash":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000
        )
    else:
        raise ValueError(f"Not support model: {model_name}")
    

def get_hf_llm(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens: int = 1024,
    **kwargs
) -> Any:
    '''
        Load free LLM model from Hugging Face
    '''
    try:
        # Try to download model if not exists
        # download_model_if_not_exists(model_name)

        # Load model with offline mode
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True,
            local_files_only=True  # Force offline mode
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True  # Force offline mode
        )

        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto"
        )

        llm = HuggingFacePipeline(
            pipeline=model_pipeline,
            model_kwargs=kwargs
        )

        return llm
    except Exception as e:
        raise Exception(f"Failed to load model {model_name}: {str(e)}")