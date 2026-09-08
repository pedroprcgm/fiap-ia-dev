"""
Downloads a sample of MedQuAD (health questions and answers) and PubMedQA
(clinical questions and answers based on medical publications), the two
datasets suggested in the Tech Challenge brief, and normalizes both to the
same format used by the hospital's synthetic data
(see generate_synthetic_hospital_data.py).

This script depends on internet access (GitHub and Hugging Face Datasets).
If the network isn't available, it automatically falls back to a small
built-in offline sample (OFFLINE_FALLBACK_*), enough for the pipeline to
run end-to-end in an environment without external access. When running with
internet, adjust MEDQUAD_SAMPLE_SIZE / PUBMEDQA_SAMPLE_SIZE to bring in more
examples.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

MEDQUAD_SAMPLE_SIZE = 30
PUBMEDQA_SAMPLE_SIZE = 30

# A sample MedQuAD XML file (questions about diseases from NIH/Genetic and
# Rare Diseases). The full repository has several folders; we use one as a
# representative sample.
MEDQUAD_SAMPLE_URL = (
    "https://raw.githubusercontent.com/abachaa/MedQuAD/master/"
    "1_CancerGov_QA/0000001_1.xml"
)

OFFLINE_FALLBACK_MEDQUAD = [
    {"question": "O que e hipertensao arterial?",
     "answer": "E a elevacao persistente da pressao arterial acima dos valores considerados normais, "
                 "aumentando o risco de doencas cardiovasculares."},
    {"question": "Quais os sintomas do diabetes tipo 2?",
     "answer": "Sede excessiva, urina frequente, fadiga e visao turva sao sintomas comuns."},
    {"question": "Como e diagnosticada a pneumonia?",
     "answer": "Por avaliacao clinica, ausculta pulmonar e radiografia de torax, "
                 "podendo ser complementada por exames laboratoriais."},
]

OFFLINE_FALLBACK_PUBMEDQA = [
    {"question": "O uso de inibidores da ECA reduz eventos cardiovasculares em hipertensos?",
     "answer": "Sim, estudos clinicos indicam reducao de eventos cardiovasculares maiores em "
                 "pacientes hipertensos tratados com inibidores da ECA, especialmente em diabeticos."},
    {"question": "Metformina e eficaz como primeira linha no diabetes tipo 2?",
     "answer": "Sim, a metformina e recomendada como terapia de primeira linha por seu perfil de "
                 "seguranca e eficacia no controle glicemico."},
    {"question": "Antibioticoterapia empirica antecipada melhora desfechos em pneumonia grave?",
     "answer": "Sim, o inicio precoce da antibioticoterapia empirica esta associado a reducao de "
                 "mortalidade em pneumonia adquirida na comunidade grave."},
]


def _parse_medquad_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    pairs = []
    for qa_pair in root.iter("QAPair"):
        question_el = qa_pair.find("Question")
        answer_el = qa_pair.find("Answer")
        if question_el is not None and answer_el is not None:
            question = (question_el.text or "").strip()
            answer = (answer_el.text or "").strip()
            if question and answer:
                pairs.append({"question": question, "answer": answer})
    return pairs


def download_medquad_sample() -> list[dict]:
    try:
        resp = requests.get(MEDQUAD_SAMPLE_URL, timeout=10)
        resp.raise_for_status()
        pairs = _parse_medquad_xml(resp.text)
        if pairs:
            print(f"MedQuAD: {len(pairs)} pares baixados de {MEDQUAD_SAMPLE_URL}")
            return pairs[:MEDQUAD_SAMPLE_SIZE]
    except Exception as exc:  # network unavailable, format changed, etc.
        print(f"[aviso] Falha ao baixar MedQuAD ({exc}); usando amostra offline embutida.")
    return OFFLINE_FALLBACK_MEDQUAD


def download_pubmedqa_sample() -> list[dict]:
    try:
        from datasets import load_dataset  # local import: only needed with internet/HF hub access

        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        pairs = []
        for item in dataset.select(range(min(PUBMEDQA_SAMPLE_SIZE, len(dataset)))):
            question = item.get("question", "").strip()
            answer = (item.get("long_answer") or "").strip()
            if question and answer:
                pairs.append({"question": question, "answer": answer})
        if pairs:
            print(f"PubMedQA: {len(pairs)} pares baixados via Hugging Face Datasets.")
            return pairs
    except Exception as exc:
        print(f"[aviso] Falha ao baixar PubMedQA ({exc}); usando amostra offline embutida.")
    return OFFLINE_FALLBACK_PUBMEDQA


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    medquad = download_medquad_sample()
    pubmedqa = download_pubmedqa_sample()

    with open(RAW_DIR / "medquad_sample.json", "w", encoding="utf-8") as f:
        json.dump(medquad, f, ensure_ascii=False, indent=2)

    with open(RAW_DIR / "pubmedqa_sample.json", "w", encoding="utf-8") as f:
        json.dump(pubmedqa, f, ensure_ascii=False, indent=2)

    print(f"Salvo: {len(medquad)} pares MedQuAD e {len(pubmedqa)} pares PubMedQA em {RAW_DIR}")


if __name__ == "__main__":
    main()
