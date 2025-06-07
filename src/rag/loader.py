import glob
from tqdm import tqdm
from typing import List, Literal, Any, Dict, Union
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def remove_non_utf8_characters(text: str) -> str:
    """
    Remove non-UTF8 characters from text
    """
    return ''.join(char for char in text if ord(char) < 128)


def load_pdf(pdf_file: Any) -> Any:
    try:
        # Disable image extraction to avoid OCR issues
        docs = PyPDFLoader(pdf_file, extract_images=False).load()
        for doc in docs:
            doc.page_content = remove_non_utf8_characters(doc.page_content)
        return docs
    except Exception as e:
        print(f"Error loading {pdf_file}: {str(e)}")
        return []


class PDFLoader:
    def __init__(self) -> None:
        pass

    def __call__(self, files: List[str], **kwargs) -> Any:
        doc_loaded = []
        total_files = len(files)

        with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
            for file in files:
                result = load_pdf(file)
                if result:  # Only extend if we got valid results
                    doc_loaded.extend(result)
                pbar.update(1)

        return doc_loaded


class TextSplitter:
    def __init__(
        self,
        seperators: List[str] = ['\n\n', '\n', ' ', ''],
        chunk_size: int = 300,
        chunk_overlap: int = 0
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=seperators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def __call__(self, documents: List[Any]) -> Any:
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(
        self,
        file_type: str = Literal["str"],
        split_kwargs: Dict[str, Any] = {
            "chunk_size": 300,
            "chunk_overlap": 0
        }
    ) -> None:
        assert file_type in ["pdf"], "file_type must be pdf"

        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")
        
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(
        self,
        pdf_files: Union[str, List[str]],
        workers: int = 1
    ) -> Any:
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        
        try:
            doc_loaded = self.doc_loader(pdf_files, workers=workers)
            if not doc_loaded:
                print("Warning: No documents were loaded successfully")
                return []
            
            doc_split = self.doc_splitter(doc_loaded)
            return doc_split
        except Exception as e:
            print(f"Error in document loading process: {str(e)}")
            return []
    
    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            if not files:
                print(f"Warning: No {self.file_type} files found in {dir_path}")
                return []
        else:
            raise ValueError("file_type must be pdf")
        
        return self.load(files, workers=workers)