import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from huggingface_hub import login

from src.endpoints import test

# Load environment variables
load_dotenv()

# Login hugging face if token is available
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HUGGINGFACE_TOKEN not set. Some models may require authentication.")

# Initial app
app = FastAPI(
    title="A Simple Project With FastAPI and RAG",
    description="Here is the firt demo",
    version="1.0.0"
)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routes
app.include_router(test.route)

# langserve routes
# add_routes(
#     app,
#     playground_type="default",
#     path="/genai"
# )