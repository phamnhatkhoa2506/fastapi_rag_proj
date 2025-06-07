from typing import Union, Any, Dict
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(
        self,
        documents: Any | None = None,
        vector_db: Union[Chroma, FAISS] = Chroma,
        embedding: Any = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    ) -> None:
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._build_db(documents)

    def _build_db(self, documents: Any) -> Chroma | Any:
        return self.vector_db.from_documents(
            documents=documents,
            embedding=self.embedding
        )
    
    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Dict = {"k": 5}
    ):
        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )