from dotenv import load_dotenv
from fastapi import FastAPI

from src.endpoints import test

# Load environment variables
load_dotenv()

app = FastAPI(
    title="A Simple Project With FastAPI and RAG",
    description="Here is the firt demo",
    version="1.0.0"
)

app.include_router(test.route)