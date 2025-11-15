from typing import List

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_files(path: str) -> List[Document]:
    """Load all PDF files from `path` and return a list of Documents."""
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only minimal metadata (`source`) with the page content.

    This reduces index size and keeps only the fields required by the app.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


def text_split(minimal_docs: List[Document]) -> List[Document]:
    """Split documents into text chunks suitable for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk


def download_embeddings() -> HuggingFaceEmbeddings:
    """Create and return a HuggingFace embedding model instance."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


# module-level convenience object (keeps original behavior)
embedding = download_embeddings()
