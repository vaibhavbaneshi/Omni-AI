import os
import json
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader
)
from common.langchain_imports import Document
from PIL import Image

class FileLoaderManager:
    """
    Centralized file loader for Omni-AI.
    Supports PDFs, Word, TXT/MD, CSV, Excel, PPTX, HTML, JSON, EPUB, Code files.
    Automatically attaches metadata for traceability.
    """

    def __init__(self, uploaded_by="system"):
        self.uploaded_by = uploaded_by

    def load(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()

        # PDF
        if ext == ".pdf":
            docs = PyPDFLoader(file_path).load()
        # Word
        elif ext == ".docx":
            docs = Docx2txtLoader(file_path).load()
        # Text / Markdown
        elif ext in [".txt", ".md"]:
            docs = TextLoader(file_path).load()
        # CSV
        elif ext == ".csv":
            docs = CSVLoader(file_path).load()
        # Excel
        elif ext in [".xls", ".xlsx"]:
            docs = UnstructuredExcelLoader(file_path).load()
        # HTML
        elif ext in [".html", ".htm"]:
            docs = UnstructuredHTMLLoader(file_path).load()
        # PowerPoint
        elif ext == ".pptx":
            docs = UnstructuredPowerPointLoader(file_path).load()
        # JSON
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs = [Document(page_content=str(data), metadata={"source": file_path})]
        # Code files
        elif ext in [".py", ".js", ".java", ".cpp", ".sql"]:
            with open(file_path, "r", encoding="utf-8") as f:
                docs = [Document(page_content=f.read(), metadata={"source": file_path, "type": "code"})]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Attach standard metadata
        for d in docs:
            d.metadata.update({
                "file_type": ext.replace(".", ""),
                "source": os.path.basename(file_path),
                "uploaded_by": self.uploaded_by,
                "timestamp": datetime.now().isoformat()
            })

        return docs