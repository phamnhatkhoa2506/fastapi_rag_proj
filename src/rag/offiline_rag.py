import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from typing import Any, List

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(self)
    
    def extract_answer(
        sefl,
        text_response: str,
        pattern: str = r"Answer:\s*(.*)"
    ) -> str:
        
        match_ = re.search(
            pattern,
            text_response,
            re.DOTALL
        )

        if match_:
            answer_text = match_.group(1).strip()
            return answer_text
        else:
            return text_response
        

class Offline_RAG:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.parser = Str_OutputParser()

    def get_chain(self, retriever: Any) -> Any:
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.parser
        )

        return rag_chain
    
    def format_docs(self, docs: List[Any]) -> str:
        return "\n\n".join(docs.page_content for doc in docs) 