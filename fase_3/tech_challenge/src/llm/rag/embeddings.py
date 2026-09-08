"""
Lightweight TF-IDF-based embedding, used instead of a neural embedding model
(e.g. sentence-transformers / InstructorEmbedding, as in the reference
material) so the RAG index can work without a GPU and without downloading
external models.

Implements LangChain's `Embeddings` interface (embed_documents/embed_query),
so it can be swapped for HuggingFaceEmbeddings/OpenAIEmbeddings in production
without changing any other module - only where the object is created in
vector_store.py.
"""
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings(Embeddings):
    """Vectorizes text with TF-IDF. The vectorizer must be fitted once over
    the full corpus before embedding individual queries, so the vocabularies
    match."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings must be fitted before embedding queries.")
        return self.vectorizer.transform([text]).toarray().tolist()[0]
