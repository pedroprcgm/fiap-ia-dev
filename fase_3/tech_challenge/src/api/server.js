/**
 * API de integracao entre o agente de IA (Python, src/llm/service) e a UI
 * (Angular, src/ui) - ver Secao 7 de
 * `docs/Documento de Especificacoes - UI do Assistente Medico`.
 *
 * Esta API e so um gateway/BFF: nao implementa nenhuma logica clinica, de
 * guardrail nem de filtragem de campos - isso tudo continua exclusivamente
 * no servico Python (src/llm/service/doctor_view.py). Aqui so validamos a
 * forma da requisicao e repassamos a chamada.
 *
 * Variaveis de ambiente:
 *   PORT            - porta da API Node (padrao 3000)
 *   LLM_SERVICE_URL - endereco do servico Python interno (padrao http://localhost:8001)
 *   UI_ORIGIN       - origem permitida para CORS (padrao http://localhost:4200)
 *
 * Rodar com: npm start
 */
const express = require("express");
const cors = require("cors");

const app = express();

const PORT = process.env.PORT || 3000;
const LLM_SERVICE_URL = process.env.LLM_SERVICE_URL || "http://localhost:8001";
const UI_ORIGIN = process.env.UI_ORIGIN || "http://localhost:4200";

app.use(cors({ origin: UI_ORIGIN }));
app.use(express.json());

async function forwardToLlmService(path, options) {
  const response = await fetch(`${LLM_SERVICE_URL}${path}`, options);
  const body = await response.json().catch(() => null);
  return { status: response.status, ok: response.ok, body };
}

// GET /patients - lista de pacientes para popular o seletor da Tela 1.
app.get("/patients", async (_req, res) => {
  try {
    const { ok, status, body } = await forwardToLlmService("/patients");
    if (!ok) {
      return res.status(status).json({ error: "Falha ao consultar o servico do agente." });
    }
    res.json(body);
  } catch (err) {
    console.error("Erro ao buscar pacientes:", err.message);
    res.status(502).json({ error: "Servico do agente indisponivel. Verifique se o servico Python (src/llm/service) esta rodando." });
  }
});

// POST /ask {patient_id, question} - devolve o DoctorView ja filtrado pelo
// servico Python (response_text, sources, pending_exam_notice). Nunca
// adiciona nem remove campos dessa resposta - ver RF07 na especificacao.
app.post("/ask", async (req, res) => {
  const { patient_id: patientId, question } = req.body || {};
  if (!patientId || !question) {
    return res.status(400).json({ error: "patient_id e question sao obrigatorios." });
  }

  try {
    const { ok, status, body } = await forwardToLlmService("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, question }),
    });
    if (!ok) {
      return res.status(status).json({ error: "Falha ao consultar o servico do agente." });
    }
    res.json(body);
  } catch (err) {
    console.error("Erro ao processar pergunta:", err.message);
    res.status(502).json({ error: "Servico do agente indisponivel. Verifique se o servico Python (src/llm/service) esta rodando." });
  }
});

app.listen(PORT, () => {
  console.log(`API Node rodando em http://localhost:${PORT} (servico do agente em ${LLM_SERVICE_URL})`);
});
