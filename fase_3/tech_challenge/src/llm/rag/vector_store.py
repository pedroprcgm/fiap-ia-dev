"""
RAG (Retrieval-Augmented Generation) index over the hospital's internal
protocols, doctor FAQs and report templates - the knowledge base that
grounds the assistant's responses, with source tracking (the challenge's
explainability requirement).

The index is rebuilt in memory on every run (ephemeral Chroma, no
persist_directory) from the files in data/raw/. For a corpus this size,
reindexing on every boot is simpler and more robust than keeping a
persisted index on disk - and it avoids locking issues in synced folders
(see the technical note in the README).
"""
import json
from pathlib import Path

from langchain_community.vectorstores import Chroma

from src.llm.rag.embeddings import TfidfEmbeddings

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw"


def _load_documents() -> tuple[list[str], list[dict]]:
    """Reads protocols, FAQs and report templates and returns (texts, metadata),
    where each metadata entry keeps the exact source used later for explainability."""
    texts = []
    metadata_list = []

    protocols_path = RAW_DIR / "protocolos_internos.json"
    if protocols_path.exists():
        for item in json.loads(protocols_path.read_text(encoding="utf-8")):
            texts.append(item["content"])
            metadata_list.append({
                "type": "internal_protocol",
                "source": item["title"],
                "specialty": item["specialty"],
            })

    faqs_path = RAW_DIR / "faqs_medicos.json"
    if faqs_path.exists():
        for item in json.loads(faqs_path.read_text(encoding="utf-8")):
            texts.append(f"{item['question']}\n{item['answer']}")
            metadata_list.append({
                "type": "medical_faq",
                "source": f"FAQ - {item['condition']}",
                "specialty": item["specialty"],
            })

    reports_path = RAW_DIR / "modelos_laudos.json"
    if reports_path.exists():
        for item in json.loads(reports_path.read_text(encoding="utf-8")):
            texts.append(item["content"])
            metadata_list.append({
                "type": "report_template",
                "source": item["title"],
                "specialty": item["specialty"],
            })

    return texts, metadata_list


class HospitalKnowledgeBase:
    def __init__(self):
        texts, metadata_list = _load_documents()
        if not texts:
            raise RuntimeError(
                "No documents found in data/raw/. Run this first:\n"
                "  python -m src.data_prep.generate_synthetic_hospital_data"
            )

        self._embeddings = TfidfEmbeddings().fit(texts)
        self._vectorstore = Chroma.from_texts(
            texts=texts, embedding=self._embeddings, metadatas=metadata_list
        )
        self._total_documents = len(texts)

    def search(self, question: str, k: int = 3) -> list[dict]:
        """Returns the k most relevant chunks, each with its exact source
        (so the assistant can cite where the information came from).

        Note: we use Chroma's native L2 distance (`similarity_search_with_score`)
        instead of `similarity_search_with_relevance_scores`, because the
        latter assumes a distance range typical of neural embeddings and
        produces values outside [0, 1] with our TF-IDF embedding. We convert
        the distance (lower = more similar) into a 0-1 score (higher = more
        similar) just to make it easier to read - the ranking is the same.
        """
        results = self._vectorstore.similarity_search_with_score(question, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "similarity": round(1 / (1 + float(distance)), 3),
            }
            for doc, distance in results
        ]

    def __len__(self):
        return self._total_documents
