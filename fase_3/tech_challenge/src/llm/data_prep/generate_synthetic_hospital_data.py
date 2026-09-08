"""
Generates the synthetic internal data of the fictional "Hospital Pos Tech",
simulating the three document types requested in the Tech Challenge:
    1) internal medical protocols;
    2) frequently asked questions (FAQs) from doctors;
    3) report, prescription and procedure templates.

Since no real hospital data exists, these documents are generated
programmatically (not by an LLM), from templates with variation by
specialty/clinical condition. This keeps the process deterministic,
auditable and easy to expand with more cases.
"""
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

CONDITIONS = [
    {
        "specialty": "Cardiologia",
        "condition": "Hipertensao Arterial Sistemica",
        "symptoms": "cefaleia, tontura, pressao arterial >= 140/90 mmHg em duas aferições",
        "exams": ["Eletrocardiograma", "Perfil lipidico", "Funcao renal (ureia/creatinina)"],
        "treatment_plan": "Reforcar mudanca de estilo de vida (reducao de sodio, atividade fisica) e iniciar "
                   "inibidor da ECA em dose baixa, reavaliando em 4 semanas.",
    },
    {
        "specialty": "Endocrinologia",
        "condition": "Diabetes Mellitus tipo 2",
        "symptoms": "poliuria, polidipsia, glicemia de jejum >= 126 mg/dL em duas ocasioes",
        "exams": ["Glicemia de jejum", "Hemoglobina glicada (HbA1c)", "Funcao renal"],
        "treatment_plan": "Iniciar metformina, orientacao nutricional e reavaliacao de HbA1c em 90 dias.",
    },
    {
        "specialty": "Pneumologia",
        "condition": "Pneumonia Adquirida na Comunidade",
        "symptoms": "febre, tosse produtiva, dispneia, estertores localizados na ausculta",
        "exams": ["Radiografia de torax", "Hemograma completo", "Proteina C reativa"],
        "treatment_plan": "Classificar gravidade (escore CURB-65); iniciar antibioticoterapia empirica "
                   "conforme protocolo institucional e reavaliar em 48-72h.",
    },
    {
        "specialty": "Gastroenterologia",
        "condition": "Doenca do Refluxo Gastroesofagico",
        "symptoms": "pirose, regurgitacao acida, piora pos-prandial e ao deitar",
        "exams": ["Endoscopia digestiva alta (se sinais de alarme)", "Nenhum exame se caso tipico"],
        "treatment_plan": "Orientar medidas antirrefluxo e iniciar inibidor de bomba de protons por 4-8 semanas.",
    },
    {
        "specialty": "Ortopedia",
        "condition": "Lombalgia Mecanica Aguda",
        "symptoms": "dor lombar sem irradiacao, sem sinais de alarme (febre, perda de peso, deficit neurologico)",
        "exams": ["Nenhum exame de imagem na fase aguda sem sinais de alarme"],
        "treatment_plan": "Analgesia, orientacao de manutencao de atividade e retorno se sinais de alarme "
                   "ou ausencia de melhora em 4-6 semanas.",
    },
    {
        "specialty": "Infectologia",
        "condition": "Infeccao do Trato Urinario nao complicada",
        "symptoms": "disuria, polaciuria, urgencia miccional, sem febre",
        "exams": ["Urina tipo I (EAS)", "Urocultura se recorrencia ou falha terapeutica"],
        "treatment_plan": "Antibioticoterapia empirica de curta duracao conforme protocolo institucional "
                   "e resistencia bacteriana local.",
    },
]

# Condições cujos "próximos passos" dependem do estágio da doença - em vez de
# um único par sintomas/exames/conduta, cada caso tem uma lista de estágios,
# e geramos um protocolo combinado e uma FAQ por estágio (perguntas do tipo
# "quais os proximos passos para uma pessoa com X em estagio Y?").
STAGED_CONDITIONS = [
    {
        "specialty": "Oncologia",
        "condition": "Cancer de Mama",
        "stages": [
            {
                "label": "inicial",
                "description": "estagios I e II, doenca localizada",
                "symptoms": "nodulo mamario palpavel ou achado em mamografia de rastreamento, "
                            "sem sinais clinicos de disseminacao a distancia",
                "exams": [
                    "Mamografia diagnostica",
                    "Ultrassonografia mamaria",
                    "Biopsia percutanea (core biopsy)",
                    "Avaliacao de receptores hormonais (RE/RP) e HER2",
                ],
                "treatment_plan": "Encaminhar para avaliacao cirurgica (cirurgia conservadora ou "
                                  "mastectomia conforme extensao e preferencia da paciente) com "
                                  "pesquisa de linfonodo sentinela, seguida de avaliacao de terapia "
                                  "adjuvante (quimioterapia, hormonioterapia e/ou radioterapia "
                                  "conforme perfil do tumor) em conjunto com a equipe de oncologia.",
            },
            {
                "label": "medio",
                "description": "estagio III, doenca localmente avancada",
                "symptoms": "tumor de maior dimensao, possivel comprometimento de linfonodos "
                            "axilares, pele ou parede toracica, sem metastase a distancia confirmada",
                "exams": [
                    "Estadiamento completo (tomografia de torax e abdome, cintilografia ossea)",
                    "Biopsia de linfonodo suspeito",
                    "Reavaliacao de receptores hormonais e HER2",
                ],
                "treatment_plan": "Iniciar quimioterapia neoadjuvante para reducao do tumor, com "
                                  "reavaliacao cirurgica apos resposta ao tratamento; radioterapia "
                                  "complementar conforme resposta e caracteristicas do tumor; caso "
                                  "discutido em junta multidisciplinar de oncologia antes de definir "
                                  "a sequencia terapeutica.",
            },
            {
                "label": "avancado",
                "description": "estagio IV, doenca metastatica",
                "symptoms": "doenca com metastase confirmada (ossea, pulmonar, hepatica, cerebral "
                            "ou outros sitios), podendo cursar com sintomas sistemicos como dor "
                            "ossea, perda de peso e fadiga",
                "exams": [
                    "Exames de imagem para mapeamento completo de metastases (tomografia, "
                    "cintilografia ossea, ressonancia conforme sitio)",
                    "Biopsia do sitio metastatico quando acessivel",
                    "Reavaliacao de biomarcadores para direcionar terapia sistemica",
                ],
                "treatment_plan": "Priorizar tratamento sistemico com intuito de controle da doenca "
                                  "e qualidade de vida (quimioterapia, terapia-alvo e/ou "
                                  "hormonioterapia conforme perfil tumoral), associado a cuidados de "
                                  "suporte para controle de sintomas; discutir precocemente com a "
                                  "paciente e familia a incorporacao de cuidados paliativos, sem "
                                  "prejuizo do tratamento oncologico ativo.",
            },
        ],
    },
]


def generate_protocol(case: dict) -> dict:
    text = (
        f"PROTOCOLO CLINICO INTERNO - {case['specialty'].upper()}\n"
        f"Condicao: {case['condition']}\n\n"
        f"1. Quadro clinico sugestivo: {case['symptoms']}.\n"
        f"2. Exames complementares recomendados: {', '.join(case['exams'])}.\n"
        f"3. Conduta preconizada pelo hospital: {case['treatment_plan']}\n"
        f"4. Este protocolo e uma sugestao institucional e nao substitui o julgamento "
        f"clinico do medico assistente nem a avaliacao individual do paciente.\n"
    )
    return {
        "type": "protocol",
        "specialty": case["specialty"],
        "condition": case["condition"],
        "title": f"Protocolo interno - {case['condition']}",
        "content": text,
    }


def generate_faq(case: dict) -> dict:
    question = f"Qual a conduta preconizada pelo hospital para {case['condition']}?"
    answer = (
        f"Para {case['condition']} ({case['specialty']}), o protocolo interno recomenda: "
        f"{case['treatment_plan']} Exames de apoio sugeridos: {', '.join(case['exams'])}."
    )
    return {
        "type": "faq",
        "specialty": case["specialty"],
        "condition": case["condition"],
        "question": question,
        "answer": answer,
    }


def generate_report_template(case: dict) -> dict:
    text = (
        f"MODELO DE LAUDO - {case['specialty'].upper()}\n"
        f"Hipotese diagnostica: {case['condition']}\n"
        f"Achados compativeis: {case['symptoms']}.\n"
        f"Exames solicitados: {', '.join(case['exams'])}.\n"
        f"Conduta sugerida: {case['treatment_plan']}\n"
        f"Observacao: laudo-modelo de uso interno, os campos devem ser preenchidos e "
        f"validados pelo medico responsavel antes de qualquer emissao.\n"
    )
    return {
        "type": "report_template",
        "specialty": case["specialty"],
        "condition": case["condition"],
        "title": f"Modelo de laudo - {case['condition']}",
        "content": text,
    }


def generate_staged_protocol(case: dict) -> dict:
    """One combined protocol document covering every stage of a condition
    (e.g. cancer staging), instead of a single symptoms/exams/conduct block."""
    sections = []
    for stage in case["stages"]:
        sections.append(
            f"ESTAGIO {stage['label'].upper()} ({stage['description']})\n"
            f"- Quadro clinico sugestivo: {stage['symptoms']}.\n"
            f"- Exames complementares recomendados: {', '.join(stage['exams'])}.\n"
            f"- Proximos passos / conduta preconizada pelo hospital: {stage['treatment_plan']}\n"
        )
    text = (
        f"PROTOCOLO CLINICO INTERNO - {case['specialty'].upper()}\n"
        f"Condicao: {case['condition']} (conduta varia conforme estagio)\n\n"
        + "\n".join(sections)
        + "\nEste protocolo e uma sugestao institucional e nao substitui o julgamento "
        "clinico do medico assistente nem a avaliacao individual do paciente.\n"
    )
    return {
        "type": "protocol",
        "specialty": case["specialty"],
        "condition": case["condition"],
        "title": f"Protocolo interno - {case['condition']} (por estagio)",
        "content": text,
    }


def generate_staged_faqs(case: dict) -> list[dict]:
    """One FAQ per stage, answering 'quais os proximos passos para uma
    pessoa com <condicao> detectada em estagio <X>?'."""
    faqs = []
    for stage in case["stages"]:
        condition_label = f"{case['condition']} - Estagio {stage['label'].capitalize()}"
        question = (
            f"Quais os proximos passos para uma pessoa com {case['condition']} "
            f"detectado em estagio {stage['label']}?"
        )
        answer = (
            f"Para {case['condition']} em estagio {stage['label']} ({stage['description']}), "
            f"o protocolo interno recomenda: {stage['treatment_plan']} "
            f"Exames de apoio sugeridos: {', '.join(stage['exams'])}."
        )
        faqs.append({
            "type": "faq",
            "specialty": case["specialty"],
            "condition": condition_label,
            "question": question,
            "answer": answer,
        })
    return faqs


def generate_staged_report_template(case: dict) -> dict:
    findings = "; ".join(
        f"estagio {stage['label']} ({stage['description']}): {stage['symptoms']}"
        for stage in case["stages"]
    )
    text = (
        f"MODELO DE LAUDO - {case['specialty'].upper()}\n"
        f"Hipotese diagnostica: {case['condition']}\n"
        f"Achados compativeis por estagio: {findings}.\n"
        f"Observacao: a conduta e os exames complementares dependem do estagio confirmado - "
        f"consultar o protocolo interno correspondente. Laudo-modelo de uso interno, os campos "
        f"devem ser preenchidos e validados pelo medico responsavel antes de qualquer emissao.\n"
    )
    return {
        "type": "report_template",
        "specialty": case["specialty"],
        "condition": case["condition"],
        "title": f"Modelo de laudo - {case['condition']} (por estagio)",
        "content": text,
    }


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    protocols = [generate_protocol(c) for c in CONDITIONS] + [
        generate_staged_protocol(c) for c in STAGED_CONDITIONS
    ]
    faqs = [generate_faq(c) for c in CONDITIONS]
    for c in STAGED_CONDITIONS:
        faqs.extend(generate_staged_faqs(c))
    report_templates = [generate_report_template(c) for c in CONDITIONS] + [
        generate_staged_report_template(c) for c in STAGED_CONDITIONS
    ]

    with open(RAW_DIR / "protocolos_internos.json", "w", encoding="utf-8") as f:
        json.dump(protocols, f, ensure_ascii=False, indent=2)

    with open(RAW_DIR / "faqs_medicos.json", "w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)

    with open(RAW_DIR / "modelos_laudos.json", "w", encoding="utf-8") as f:
        json.dump(report_templates, f, ensure_ascii=False, indent=2)

    print(f"Gerados {len(protocols)} protocolos, {len(faqs)} FAQs e {len(report_templates)} modelos de laudo em {RAW_DIR}")


if __name__ == "__main__":
    main()
