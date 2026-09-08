"""
Prompt templates and output parser for the assistant's response, following
the structured Output Parsers pattern (JSON via Pydantic) taught in
Aula 01 - Introducao a LangChain of the reference material.

The structured output is what lets the rest of the pipeline (guardrails,
logging) programmatically inspect the source and confidence level of the
response, instead of dealing with free text.
"""
# Note: in the course's reference material (langchain 0.2.x) this import is
# `from langchain.output_parsers import PydanticOutputParser`. Starting with
# langchain 0.3/1.x this parser moved to langchain_core - same class,
# updated import path.
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    response_text: str = Field(description="Sugestao de conduta ou resposta a duvida do medico")
    sources: list[str] = Field(description="Titulos dos documentos/protocolos usados para fundamentar a resposta")
    confidence_level: float = Field(description="Confianca estimada da resposta, de 0 a 1")
    requires_human_validation: bool = Field(
        default=True,
        description="Sempre true para sugestoes de conduta clinica - o assistente nunca decide sozinho",
    )


response_parser = PydanticOutputParser(pydantic_object=AssistantResponse)


MEDICAL_ASSISTANT_TEMPLATE = PromptTemplate(
    template=(
        "Voce e um assistente clinico do hospital. Responda a pergunta do medico "
        "usando SOMENTE as informacoes do contexto abaixo (protocolos internos e "
        "dados do paciente). Se o contexto nao for suficiente, diga isso "
        "explicitamente em vez de inventar uma resposta.\n\n"
        "Contexto do paciente:\n{patient_context}\n\n"
        "Protocolos e FAQs relevantes recuperados:\n{rag_context}\n\n"
        "Pergunta do medico:\n{question}\n\n"
        "{format_instructions}\n"
    ),
    input_variables=["patient_context", "rag_context", "question"],
    partial_variables={"format_instructions": response_parser.get_format_instructions()},
)


def build_prompt(patient_context: str, rag_context: str, question: str) -> str:
    return MEDICAL_ASSISTANT_TEMPLATE.format(
        patient_context=patient_context,
        rag_context=rag_context,
        question=question,
    )
