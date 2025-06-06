from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI


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