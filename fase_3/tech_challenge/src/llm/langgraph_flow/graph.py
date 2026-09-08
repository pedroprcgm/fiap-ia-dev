"""
Builds the medical assistant's StateGraph - the central orchestration
requested by the challenge ("automated and safe decision flows...
coordinated with LangChain/LangGraph"), following the pattern taught in
Aulas 01-04 of the course's LangGraph module (StateGraph, nodes, edges,
shared state).

Flow:
    Exam Verifier -> Context (RAG) -> Treatment Suggestion
        -> Guardrails -> Alerts / Log -> END
"""
from langgraph.graph import END, StateGraph

from src.llm.langgraph_flow.nodes import (
    alerts_log_node,
    exam_verifier_node,
    guardrails_node,
    rag_context_node,
    treatment_suggestion_node,
)
from src.llm.langgraph_flow.state import AssistantState


def build_graph():
    graph = StateGraph(AssistantState)

    graph.add_node("exam_verifier", exam_verifier_node)
    graph.add_node("rag_context", rag_context_node)
    graph.add_node("treatment_suggestion", treatment_suggestion_node)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("alerts_log", alerts_log_node)

    graph.set_entry_point("exam_verifier")
    graph.add_edge("exam_verifier", "rag_context")
    graph.add_edge("rag_context", "treatment_suggestion")
    graph.add_edge("treatment_suggestion", "guardrails")
    graph.add_edge("guardrails", "alerts_log")
    graph.add_edge("alerts_log", END)

    return graph.compile()
