from pydantic import BaseModel, Field
from typing import Any
from src.rag.loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offiline_rag import Offline_RAG
from src.models.base.llm import load_llm


class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")


class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")


def build_rag_chain(
    llm: Any, 
    data_dir: str,
    data_type: str
):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(documents=doc_loaded).get_retriever()
    rag_chain = Offline_RAG(llm).get_chain(retriever)

    return rag_chain


biology_docsc = "./data_sources/biology"
bio_chain = build_rag_chain(
    load_llm(),
    data_dir=biology_docsc,
    data_type="pdf"
)