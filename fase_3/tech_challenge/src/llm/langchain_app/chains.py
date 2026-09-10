"""
Main assistant chain: pipes patient context retrieval -> RAG context
retrieval -> prompt construction -> domain LLM call -> response
structuring. This is the "pipeline that integrates the custom LLM"
requested in requirement 2 of the challenge (equivalent to the Sequential
Chains seen in Aula 04 - Chains of the reference material).

This chain is used both standalone (tests/CLI) and inside the "Treatment
Suggestion" node of the LangGraph graph (src/langgraph_flow/nodes.py).
"""
# Allows `Type | None` (PEP 604) on Python 3.9, which only supports that
# syntax natively from 3.10 onward.
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.langchain_app.document_loaders import load_medical_record_text, load_patient
from src.llm.langchain_app.prompts import AssistantResponse, response_parser
from src.llm.models.domain_llm import DomainLLM
from src.llm.rag.embeddings import PORTUGUESE_STOPWORDS
from src.llm.rag.vector_store import HospitalKnowledgeBase

BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset_fine_tuning.jsonl"


class _FinetuningOnlyMatcher:
    """Recognizes when a question is about something RAG has zero real
    documents for, so `MedicalAssistantChain.invoke` can skip the RAG search
    for it instead of risking a misleading "closest" match.

    Built directly from the `in_rag` flag in
    data/processed/dataset_fine_tuning.jsonl (see
    build_fine_tuning_dataset.py's module docstring for what that flag means
    and why) - not a hardcoded condition list. This matters because TF-IDF
    search never says "nothing relevant found": it always returns its k
    nearest documents, even when none actually relate to the question - a
    general question (no patient to ground the query) about "crise
    asmatica" - deliberately absent from data/raw/, so the model can learn
    something RAG doesn't already know - once surfaced a "Cancer de Mama"
    FAQ as the top RAG hit (shared generic words, not real relevance),
    which the demo backend then trusted and returned verbatim as the
    "answer" (see domain_llm.py::_generate_with_demo_index).

    Being data-driven means adding a new FINE_TUNING_ONLY_CONDITIONS entry
    (or any other in_rag=False example) needs no change here - just
    regenerate the dataset (`python -m src.llm.data_prep.build_fine_tuning_dataset`).

    Matches against `input + " " + output` (question AND answer content),
    not just the question. This was not the first thing tried - matching on
    questions alone kept false-positiving on any question phrased with the
    same instruction template as FINE_TUNING_ONLY_CONDITIONS's alt_question
    ("Quais os proximos passos para um paciente com X?"): that template
    isn't used anywhere else in the dataset, so it's the single rarest
    n-gram sequence in the whole corpus, and its own words dominated the
    similarity score regardless of which condition X actually was -
    "...com Diabetes Mellitus tipo 2?" and "...com Cancer de Mama?" both
    false-matched it despite sharing only the template, not the condition.
    Appending each example's answer text fixes this: answers are long,
    condition-specific, and never share an instruction template with
    anything, so real medical content dominates the match instead of
    boilerplate question phrasing.
    """

    MIN_SIMILARITY = 0.3

    def __init__(self, dataset_path: Path | None = None):
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

        path = dataset_path or DATASET_PATH
        if not path.exists():
            return

        all_texts = []
        in_rag_flags = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                all_texts.append(f"{example['input']} {example['output']}")
                in_rag_flags.append(example.get("in_rag", True))

        if not any(not flag for flag in in_rag_flags):
            return
        # Same tuning as HospitalKnowledgeBase's embeddings.py, and for the
        # same reason: several in_rag=False examples reuse the exact same
        # instruction templates as in_rag=True ones ("Qual o protocolo
        # interno para X?"). strip_accents so "asmatica"/"asmática" compare
        # equal either way; sublinear_tf so one very long answer doesn't
        # dominate purely on raw term counts.
        #
        # Important: the vectorizer is fit on the WHOLE dataset (both
        # in_rag values), not just the in_rag=False slice - fitting on just
        # that slice was tried first and made things worse, not better: with
        # only a handful of in_rag=False examples, template phrases looked
        # artificially rare/distinctive (IDF is relative to whatever corpus
        # it's fit on), so ANY question using the same template scored a
        # false match. Fitting on the full dataset gives IDF an accurate
        # picture of how common that phrase really is, correctly discounting
        # it; only the *candidates* being matched against are restricted to
        # the in_rag=False rows.
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=PORTUGUESE_STOPWORDS,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        full_matrix = self._vectorizer.fit_transform(all_texts)
        candidate_rows = [i for i, flag in enumerate(in_rag_flags) if not flag]
        self._matrix = full_matrix[candidate_rows]

    def matches(self, question: str) -> bool:
        if self._matrix is None:
            return False
        query_vector = self._vectorizer.transform([question])
        best_similarity = float(cosine_similarity(query_vector, self._matrix).max())
        return best_similarity >= self.MIN_SIMILARITY


class MedicalAssistantChain:
    def __init__(self, domain_llm: DomainLLM | None = None, knowledge_base: HospitalKnowledgeBase | None = None):
        self.domain_llm = domain_llm or DomainLLM()
        self.knowledge_base = knowledge_base or HospitalKnowledgeBase()
        self._finetuning_only_matcher = _FinetuningOnlyMatcher()

    def _patient_record_document(self, patient_id: str) -> dict | None:
        """Loads the patient's own medical record (data/processed/prontuarios/<id>.txt)
        via a LangChain TextLoader (document_loaders.py::load_medical_record_text)
        and returns it in the same shape as HospitalKnowledgeBase.search()'s
        results, so it can be merged into `rag_documents` in `invoke()` below
        and treated as just another grounding source: shown in the context
        with its own "[Fonte: ...]" tag and listed among the response's cited
        sources (see `_structure_response`), the same explainability path
        every other document in this project goes through.

        Unlike the protocols/FAQs retrieved by `HospitalKnowledgeBase`, this
        isn't a lexical search match - it's always exactly the right document
        for this consult (there's no ambiguity about which patient is being
        asked about), so similarity is fixed at 1.0 rather than computed.
        """
        document = load_medical_record_text(patient_id)
        if document is None:
            return None
        return {
            "content": document.page_content,
            "source": f"Prontuario - {patient_id}",
            "type": "patient_medical_record",
            "similarity": 1.0,
        }

    def _structure_response(self, generated_text: str, rag_sources: list[dict]) -> AssistantResponse:
        """Tries to parse the LLM output as structured JSON (the format
        requested in the prompt). If the LLM doesn't follow the format -
        the case for the demo backend, which isn't an instruction-tuned
        model - builds the structure in code from the text and the RAG
        sources."""
        try:
            return response_parser.parse(generated_text)
        except Exception:
            confidence_scores = [f["similarity"] for f in rag_sources] or [0.0]
            return AssistantResponse(
                response_text=generated_text,
                sources=[f["source"] for f in rag_sources] or ["dataset de fine-tuning (sem fonte RAG associada)"],
                confidence_level=round(sum(confidence_scores) / len(confidence_scores), 3),
                requires_human_validation=True,
            )

    def invoke(self, patient_id: str | None, question: str, k_documents: int = 3) -> AssistantResponse:
        """`patient_id` is optional: the doctor can ask a general question
        with no patient selected (e.g. "qual o protocolo para X?"), in which
        case every patient-specific lookup below is simply skipped and the
        response relies only on RAG (see src/llm/service/app.py::AskRequest,
        where this is None by default)."""
        main_condition = None
        if patient_id:
            patient = load_patient(patient_id)
            main_condition = patient["main_condition"] if patient else None

        # Grounds the RAG search in the patient's known condition, not just
        # the doctor's free-text question. This matters a lot with TF-IDF
        # (lexical, not semantic) retrieval: a vague question like "qual o
        # tratamento para o caso?" shares no distinctive vocabulary with any
        # protocol, so without this the top match is essentially noise - we
        # saw this retrieve a Cancer de Mama FAQ for a UTI patient before
        # this fix. Appending the condition name reliably steers TF-IDF
        # toward the right document.
        search_query = f"{question} {main_condition}" if main_condition else question
        if self._finetuning_only_matcher.matches(question):
            # RAG genuinely has nothing here - see _FinetuningOnlyMatcher's docstring.
            rag_documents = []
        else:
            rag_documents = self.knowledge_base.search(search_query, k=k_documents)

        # The patient's own medical record joins the RAG documents (not a
        # separate context prefix like before) - it's grounding for this
        # response exactly like a retrieved protocol/FAQ is, and should be
        # citable/explainable the same way (see `_patient_record_document`).
        # Appended AFTER the lexically-retrieved documents, not before: the
        # demo backend (_generate_with_demo_index) naively trusts whichever
        # chunk comes first as "the answer" - putting the patient's raw
        # record first would make it echo patient metadata back instead of
        # an actual protocol/FAQ for any question. The real (LoRA/mlx)
        # backends read the whole context regardless of order, so this only
        # matters for the demo backend.
        if patient_id:
            record_document = self._patient_record_document(patient_id)
            if record_document is not None:
                rag_documents = rag_documents + [record_document]

        rag_context = "\n\n".join(
            f"[Fonte: {d['source']}]\n{d['content']}" for d in rag_documents
        )

        # Patient's main condition goes before the RAG chunks, not mixed into
        # them - _generate_with_demo_index only looks at the first "[Fonte:"
        # marker onward, so this prefix is safely ignored by that backend
        # while still reaching the real (LoRA/mlx) backends, which read the
        # whole context string.
        context_parts = []
        if main_condition:
            context_parts.append(f"Condicao principal do paciente: {main_condition}")
        context_parts.append(rag_context)
        combined_context = "\n\n".join(context_parts)

        # The instruction/input here follow the same format used in the
        # fine-tuning dataset (see src/data_prep/build_fine_tuning_dataset.py),
        # so DomainLLM recognizes the pattern even on the demo backend.
        instruction = "Responda como assistente clinico do hospital, com base no protocolo interno."
        generated_text = self.domain_llm.generate_response(instruction, question, context=combined_context)

        return self._structure_response(generated_text, rag_documents)
