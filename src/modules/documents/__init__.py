from src.modules.documents.dependencies import (
    DocumentsRepositoryDep,
    get_documents_repository,
)
from src.modules.documents.models import Document
from src.modules.documents.repository import DocumentsRepository

__all__ = [
    "Document",
    "DocumentsRepository",
    "DocumentsRepositoryDep",
    "get_documents_repository",
]

