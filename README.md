# Pipeline para análise de tumor com classifição binária

Classificação de câncer de mama (trabalho FIAP) com SVC linear, regressão linear e visualização de importância de features via SHAP.


## Apresentação em vídeo

[Assista ao vídeo explicando os resultados obtidos e o relatório técnico](https://www.youtube.com/watch?v=2Yq7G217B7U)

## Conteúdo

- `cancer_classification.ipynb` — notebook principal
- `data.csv` — dataset
- `Dockerfile`, `docker-compose.yml` — execução em container

## Pré-requisitos

- Docker Desktop (macOS/Windows) ou Docker Engine + Docker Compose v2 (Linux)

## Modo interativo (JupyterLab)

```bash
cd breast_cancer_v2
docker compose up jupyter
```

Abra `http://localhost:8888/lab` no navegador. O diretório do projeto está montado em `/home/jovyan/work`, então as edições feitas no JupyterLab são salvas direto nos arquivos do host.

Para parar: `Ctrl+C` no terminal, ou `docker compose down` em outro terminal.

> **Aviso:** o token de autenticação do Jupyter está desabilitado para uso local acadêmico. Não exponha a porta 8888 para a rede.

## Modo headless (executar o notebook e gerar HTML)

```bash
cd breast_cancer_v2
docker compose --profile run run --rm notebook-run
```

Saídas geradas no próprio diretório:

- `cancer_classification.executed.ipynb` — notebook com todas as células executadas
- `cancer_classification.executed.html` — versão HTML para distribuição

Esses dois arquivos são ignorados pelo git (ver `.gitignore` na raiz).

## Porta 8888 ocupada?

Se a porta 8888 já estiver em uso no host, crie um `docker-compose.override.yml` ao lado do `docker-compose.yml`:

```yaml
services:
  jupyter:
    ports:
      - "8889:8888"
```

E acesse `http://localhost:8889/lab`.

## Reprodutibilidade

A imagem base está fixada em `quay.io/jupyter/scipy-notebook:2026-04-27`. Para atualizar, edite o `Dockerfile` e refaça o build com `docker compose build`.
