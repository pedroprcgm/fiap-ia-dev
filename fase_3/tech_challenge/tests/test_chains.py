import pytest

from src.llm.langchain_app.chains import MedicalAssistantChain
from src.llm.langchain_app.document_loaders import list_patients, load_patient


def _patient_id_for_condition(condition: str) -> str:
    for patient in list_patients():
        if patient["main_condition"] == condition:
            return patient["patient_id"]
    pytest.skip(f"Nenhum paciente sintetico com condicao '{condition}' encontrado.")


def test_load_patient_returns_the_matching_record():
    patients = list_patients()
    assert patients, "Base de pacientes vazia - rode os scripts de data_prep."
    first = patients[0]

    loaded = load_patient(first["patient_id"])

    assert loaded is not None
    assert loaded["patient_id"] == first["patient_id"]
    assert loaded["main_condition"] == first["main_condition"]


def test_load_patient_returns_none_for_unknown_id():
    assert load_patient("PACIENTE-INEXISTENTE") is None


def test_vague_question_still_retrieves_documents_for_the_patients_own_condition():
    """Regression test for a real bug: a vague question ("qual o tratamento
    para o caso?") with no mention of the condition used to retrieve
    completely unrelated documents (a UTI patient got a Cancer de Mama FAQ
    back), because RAG search only used the raw question text - see
    MedicalAssistantChain.invoke, which now grounds the search query in the
    patient's own main_condition."""
    condition = "Infeccao do Trato Urinario nao complicada"
    patient_id = _patient_id_for_condition(condition)

    chain = MedicalAssistantChain()
    response = chain.invoke(patient_id, "qual o tratamento para o caso?")

    assert response.sources, "Resposta deveria citar ao menos uma fonte."
    assert any("Urinario" in source or "Urinaria" in source for source in response.sources), (
        f"Fontes retornadas nao mencionam a condicao do paciente: {response.sources}"
    )
    assert not any("Mama" in source for source in response.sources), (
        f"Bug reproduzido: fonte sobre Cancer de Mama retornada para paciente com {condition}: "
        f"{response.sources}"
    )
