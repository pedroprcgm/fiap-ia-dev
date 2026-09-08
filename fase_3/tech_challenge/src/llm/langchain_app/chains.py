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

from src.llm.langchain_app.document_loaders import load_medical_record_history, load_patient
from src.llm.langchain_app.prompts import AssistantResponse, response_parser
from src.llm.models.domain_llm import DomainLLM
from src.llm.rag.vector_store import HospitalKnowledgeBase


class MedicalAssistantChain:
    def __init__(self, domain_llm: DomainLLM | None = None, knowledge_base: HospitalKnowledgeBase | None = None):
        self.domain_llm = domain_llm or DomainLLM()
        self.knowledge_base = knowledge_base or HospitalKnowledgeBase()

    def _patient_context(self, patient_id: str) -> str:
        events = load_medical_record_history(patient_id)
        if not events:
            return "Sem historico de prontuario disponivel para este paciente."
        return "\n".join(f"- {e['event_date']}: {e['description']}" for e in events)

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

    def invoke(self, patient_id: str, question: str, k_documents: int = 3) -> AssistantResponse:
        patient = load_patient(patient_id)
        main_condition = patient["main_condition"] if patient else None
        patient_context = self._patient_context(patient_id)

        # Grounds the RAG search in the patient's known condition, not just
        # the doctor's free-text question. This matters a lot with TF-IDF
        # (lexical, not semantic) retrieval: a vague question like "qual o
        # tratamento para o caso?" shares no distinctive vocabulary with any
        # protocol, so without this the top match is essentially noise - we
        # saw this retrieve a Cancer de Mama FAQ for a UTI patient before
        # this fix. Appending the condition name reliably steers TF-IDF
        # toward the right document.
        search_query = f"{question} {main_condition}" if main_condition else question
        rag_documents = self.knowledge_base.search(search_query, k=k_documents)
        rag_context = "\n\n".join(
            f"[Fonte: {d['source']}]\n{d['content']}" for d in rag_documents
        )

        # Patient context (condition + record history) goes before the RAG
        # chunks, not mixed into them - _generate_with_demo_index only looks
        # at the first "[Fonte:" marker onward, so this prefix is safely
        # ignored by that backend while still reaching the real (LoRA/mlx)
        # backends, which read the whole context string.
        context_parts = []
        if main_condition:
            context_parts.append(f"Condicao principal do paciente: {main_condition}")
        if patient_context:
            context_parts.append(f"Historico do prontuario:\n{patient_context}")
        context_parts.append(rag_context)
        combined_context = "\n\n".join(context_parts)

        # The instruction/input here follow the same format used in the
        # fine-tuning dataset (see src/data_prep/build_fine_tuning_dataset.py),
        # so DomainLLM recognizes the pattern even on the demo backend.
        instruction = "Responda como assistente clinico do hospital, com base no protocolo interno."
        generated_text = self.domain_llm.generate_response(instruction, question, context=combined_context)

        return self._structure_response(generated_text, rag_documents)
