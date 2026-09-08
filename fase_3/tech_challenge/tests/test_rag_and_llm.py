import pytest

from src.llm.models.domain_llm import DomainLLM
from src.llm.rag.vector_store import HospitalKnowledgeBase


@pytest.fixture(scope="module")
def knowledge_base():
    return HospitalKnowledgeBase()


@pytest.fixture(scope="module")
def domain_llm():
    return DomainLLM()


def test_knowledge_base_indexes_documents(knowledge_base):
    assert len(knowledge_base) > 0


def test_search_retrieves_relevant_source(knowledge_base):
    results = knowledge_base.search(
        "Qual o protocolo interno para Hipertensao Arterial Sistemica?", k=1
    )
    assert len(results) == 1
    assert "Hipertensao" in results[0]["source"]
    assert results[0]["similarity"] > 0


def test_domain_llm_generates_non_empty_response(domain_llm):
    response = domain_llm.generate_response(
        "Responda como assistente clinico do hospital, com base no protocolo interno.",
        "Qual o protocolo interno para Diabetes Mellitus tipo 2?",
    )
    assert isinstance(response, str)
    assert len(response) > 0


def test_demo_backend_ignores_patient_context_prefix_before_rag_sources(domain_llm):
    """MedicalAssistantChain.invoke prepends patient info (condition, record
    history) before the RAG "[Fonte: ...]" chunks in the context string (see
    chains.py). The demo/retrieval backend must still pick the first RAG
    chunk correctly instead of mistaking that prefix for one."""
    rag_context = "[Fonte: Protocolo X]\nConteudo do protocolo X."
    context_with_patient_prefix = (
        "Condicao principal do paciente: Alguma Condicao\n\n"
        "Historico do prontuario:\nAlgum historico.\n\n" + rag_context
    )

    without_prefix = domain_llm.generate_response("instrucao", "pergunta", context=rag_context)
    with_prefix = domain_llm.generate_response("instrucao", "pergunta", context=context_with_patient_prefix)

    assert without_prefix == with_prefix
    assert "Condicao principal do paciente" not in with_prefix
