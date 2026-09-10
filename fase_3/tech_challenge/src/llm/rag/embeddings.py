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

# sklearn ships no built-in Portuguese stopword list (only English). Without
# one, generic connector/boilerplate words ("de", "para", "conforme") count
# just like real medical vocabulary, which - combined with unigrams only -
# was letting an unrelated document win a search purely on shared filler
# words. Bigrams are included too, so multi-word medical terms ("dor
# lombar", "crise asmatica") count as a unit instead of just their
# individual (very generic) words. Also reused by
# chains.py::_FinetuningOnlyMatcher, which has the exact same problem on an
# even smaller corpus.
PORTUGUESE_STOPWORDS = [
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "dos",
    "do", "e", "ela", "elas", "ele", "eles", "em", "entre", "essa", "essas",
    "esse", "esses", "esta", "estao", "este", "estes", "foi", "isso", "ja",
    "mais", "muito", "na", "nao", "nas", "no", "nos", "num", "numa", "o",
    "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "que",
    "se", "sem", "ser", "seu", "seus", "sim", "sobre", "sua", "suas",
    "um", "uma", "uns", "umas",
]


class TfidfEmbeddings(Embeddings):
    """Vectorizes text with TF-IDF. The vectorizer must be fitted once over
    the full corpus before embedding individual queries, so the vocabularies
    match."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=PORTUGUESE_STOPWORDS,
            sublinear_tf=True,
            # So "asmatica"/"asmática" (whichever form a document or a
            # doctor's question happens to use) compare equal.
            strip_accents="unicode",
        )
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
