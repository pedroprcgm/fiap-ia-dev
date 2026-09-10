import pytest

from src.llm.data_prep.build_fine_tuning_dataset import FINE_TUNING_ONLY_CONDITIONS
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


def test_invoke_accepts_no_patient_for_a_general_question():
    """patient_id is optional (see chains.py::invoke) so the doctor can ask
    a general question with no patient selected - every patient lookup
    (load_patient, medical record history) should simply be skipped instead
    of raising."""
    chain = MedicalAssistantChain()
    response = chain.invoke(None, "Qual o protocolo interno para Hipertensao Arterial Sistemica?")

    assert response.response_text
    assert response.sources


def test_general_question_about_finetuning_only_condition_skips_rag_instead_of_a_wrong_source():
    """Regression test for a real bug: a general question (no patient
    selected) about "crise asmatica" came back with a response about
    "cancer de mama". Root cause: Crise Asmatica Aguda is deliberately
    excluded from data/raw/ (FINE_TUNING_ONLY_CONDITIONS - see
    build_fine_tuning_dataset.py), so RAG has zero real documents about it;
    but TF-IDF search never says "nothing relevant found" - it always
    returns its k nearest documents, and without a patient's main_condition
    to ground the query, the top "match" was just an unrelated document
    that happened to share generic words with the question. chains.py now
    recognizes these questions and skips RAG entirely for them instead of
    citing/trusting whatever it finds."""
    condition = FINE_TUNING_ONLY_CONDITIONS[0]["condition"]
    chain = MedicalAssistantChain()

    response = chain.invoke(None, f"Qual o tratamento para {condition.lower()}?")

    assert "Mama" not in response.response_text
    assert not any("Mama" in source for source in response.sources), (
        f"Bug reproduzido: fonte nao relacionada retornada para pergunta sobre "
        f"'{condition}': {response.sources}"
    )
    # A palavra distintiva da condicao (ex.: "asmatica") deve aparecer na
    # resposta - vem do proprio backend demo, que faz sua propria busca no
    # dataset de fine-tuning (que inclui essa condicao) quando nao ha
    # contexto do RAG.
    distinctive_word = max(condition.split(), key=len)
    assert distinctive_word.lower() in response.response_text.lower()


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
