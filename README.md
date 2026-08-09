# ml3-f3-tech-challenge

Sistema de **triagem automática de laudos médicos** — classifica o texto de um
laudo/relato clínico em `normal` / `atencao` / `urgente`, servido via API REST
em container Docker. FIAP MLE — Tech Challenge Fase 3 (tema: *Deploy de
Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de
Latência*).

## Problema

Um hospital de referência precisa priorizar exames de texto assim que eles
chegam, sem depender de um médico revisar cada laudo manualmente antes da
fila de atendimento. Um classificador leve de NLP, embarcado numa API,
permite sinalizar automaticamente os casos potencialmente urgentes — o
ganho não é "substituir" o julgamento clínico, é reduzir o tempo até que um
laudo grave chegue à frente de alguém que possa agir.

## Decisão Arquitetural (Etapa 1 — Deploy em Nuvem)

**Modo de deploy: real-time, não batch.** Triagem existe para comprimir o
tempo entre "laudo chegou" e "alguém urgente foi visto" — processar em lote
(ex.: a cada hora) reintroduz exatamente o atraso que o sistema deveria
eliminar. A API precisa responder por requisição, com latência baixa e
previsível (ver seção *Otimização de Latência*).

**Provedor recomendado: AWS**, com o seguinte desenho:

| Componente | Serviço AWS | Por quê |
|---|---|---|
| Registro de imagem | ECR | Recebe o build do `Dockerfile` (stage `api`) direto do pipeline CI/CD |
| Execução do container | ECS Fargate | Serverless (sem gerenciar EC2/nós), escala horizontalmente pela carga de requisições, cobra só pelo tempo de execução — adequado a uma API cujo tráfego varia ao longo do dia hospitalar |
| Exposição/roteamento | Application Load Balancer | Health check em `/health`, TLS termination, distribui entre as tasks do Fargate |
| Métricas/observabilidade | CloudWatch Container Insights (produção) + a stack local Prometheus/Grafana deste repo (dev) | Em produção, o `/metrics` da API também pode ser raspado por um Prometheus gerenciado (Amazon Managed Service for Prometheus) sem trocar a instrumentação |
| Orquestração de retreino | Amazon MWAA (Managed Airflow) ou o mesmo `docker-compose.airflow.yml` num EC2/Fargate dedicado | Reaproveita a DAG deste repo sem reescrever a lógica de treino |

**Alternativas consideradas:**
- **Azure Container Apps / Azure ML Endpoints** — equivalente ao ECS Fargate
  em simplicidade operacional; faria sentido se o restante da infra do
  hospital já estivesse no Azure (ex.: Active Directory, PACS/RIS
  integrados).
- **GCP Cloud Run** — provavelmente a opção mais simples de todas (deploy
  direto de container, scale-to-zero); competitiva se o time já usa
  BigQuery/Vertex AI para outras cargas de ML do hospital.

A escolha por AWS/Fargate aqui é sobre onde este time já tem mais
familiaridade operacional e sobre a maturidade do Airflow gerenciado (MWAA)
— não há um requisito técnico do problema que exclua Azure ou GCP; os três
resolvem "container real-time com autoscaling" igualmente bem.

## Stack

- **Modelo**: TF-IDF + Random Forest / Logistic Regression (`scikit-learn`) — leve o
  suficiente para treinar em segundos e rodar em CPU.
- **API**: FastAPI + Uvicorn.
- **Otimização de latência**: exportação do classificador para **ONNX
  Runtime** (Etapa 4) — ver `src/optimization/export_onnx.py`.
- **CI/CD**: GitHub Actions (lint → test → build).
- **Orquestração de treino/retreino**: Apache Airflow (`docker-compose.airflow.yml`).
- **Monitoramento**: `prometheus-client` na API + Prometheus + Grafana via Docker Compose.
- **Empacotamento**: Poetry + Docker multi-stage.

## Estrutura do Projeto

```
.
├── .github/workflows/ci.yml        # pipeline CI/CD (lint → test → build)
├── airflow/
│   └── dags/triage_training_dag.py # DAG: ingestão → treino → avaliação → export ONNX
├── config/config.yaml              # config única do projeto (dados, features, modelo, artefatos)
├── data/{raw,processed,external}/  # dados brutos e splits (git-ignorados, exceto .gitkeep)
├── docker-compose.yml              # API + Prometheus + Grafana
├── docker-compose.airflow.yml      # Airflow (LocalExecutor + Postgres), sobe à parte
├── Dockerfile                      # stages: builder, train, api
├── docs/
│   ├── architecture.md             # detalhamento da decisão arquitetural
│   └── model_card.md               # ficha do modelo (dados, métricas, limitações)
├── metrics/                        # eval_metrics.json, latency_comparison.json
├── models/{artifacts,onnx}/        # pipeline sklearn (.joblib) e classificador ONNX
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/{provisioning,dashboards}/
├── scripts/
│   ├── generate_synthetic_dataset.py
│   └── measure_latency.py
├── src/
│   ├── config/settings.py          # Pydantic Settings (env vars)
│   ├── data/make_dataset.py        # carga + split train/val/test
│   ├── features/text_preprocessing.py
│   ├── models/{classifier,registry}.py
│   ├── training/trainer.py
│   ├── evaluation/evaluate.py
│   ├── optimization/export_onnx.py
│   ├── serving/{api,model_loader,metrics,schemas}.py
│   └── utils/
├── tests/
├── Makefile
└── pyproject.toml
```

## Início Rápido

```bash
# 1. Instalar Poetry e todas as dependências (prod + dev) em .venv/
make install

# 2. Configurar variáveis de ambiente
cp .env.example .env

# 3. Gerar o dataset (sintético por padrão — ver seção "Dataset")
make dataset

# 4. Treinar o modelo e avaliar
make train
make evaluate

# 5. (opcional) Otimizar latência — exporta o classificador para ONNX
make export-onnx
make benchmark-latency

# 6. Subir a API localmente
make api
# → http://localhost:8000/docs
```

### Testando a API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Paciente relata dor toracica intensa e subita, irradiando para o braco esquerdo."}'
```

## Docker

```bash
# Build de todas as imagens (api, train)
make compose-build

# Sobe API + Prometheus + Grafana
make compose-up

# Treina o modelo dentro do container (perfil "train", não sobe com `up` normal)
docker compose run --rm train
docker compose run --rm train python -m src.evaluation.evaluate
docker compose run --rm train python -m src.optimization.export_onnx
```

A API só entra em modo "pronto" (`/health` → `status: ok`) depois que houver
um modelo salvo em `models/artifacts/`. Sem modelo, ela sobe em modo
degradado (`503` em `/predict`) — isso é intencional: o container não
deveria falhar o healthcheck de infraestrutura só porque o modelo ainda não
foi treinado/promovido.

## API de Serving

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | metadados do serviço |
| GET | `/health` | estado de carregamento do modelo |
| POST | `/predict` | classifica um laudo (`{"text": "..."}` → `label`, `scores`, `backend`) |
| GET | `/metrics` | métricas no formato Prometheus |
| GET | `/docs` | Swagger UI (gerado pelo FastAPI) |

## Monitoramento

`docker compose up` sobe:

- **API** em `http://localhost:8000` (instrumentada com `prometheus-client` —
  ver `src/serving/metrics.py`: contagem de requisições por rota/status e
  histograma de latência).
- **Prometheus** em `http://localhost:9090`, raspando `api:8000/metrics` a
  cada 5s (`monitoring/prometheus/prometheus.yml`).
- **Grafana** em `http://localhost:3000` (login `admin`/`admin` por padrão —
  troque via `GRAFANA_ADMIN_PASSWORD` no `.env`), com o datasource do
  Prometheus e o dashboard `Triage API — Overview`
  (`monitoring/grafana/dashboards/triage-api-overview.json`) **já
  provisionados automaticamente**, sem passos manuais.

Painéis do dashboard: total de requisições, taxa de erro (%), latência
p50/p95 por rota, requisições por segundo por rota e distribuição das
classificações de urgência retornadas pela API.

Para gerar tráfego e ver os gráficos se populando:

```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "febre alta com confusao mental"}' > /dev/null
done
```

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` roda em todo push/PR para `main`, em 3 jobs
encadeados: **lint** (`ruff check` + `ruff format --check`) → **test**
(`pytest` com cobertura) → **build** (build da imagem Docker do stage `api`,
validando que o Dockerfile é funcional).

## Orquestração (Airflow)

```bash
make airflow-up    # sobe Postgres + Airflow (LocalExecutor) via docker-compose.airflow.yml
# UI em http://localhost:8080 (admin/admin)
make airflow-down
```

A DAG `triage_training` (`airflow/dags/triage_training_dag.py`) encadeia:

```
ingest_data → train_model → evaluate_model → export_onnx
```

Cada task chama diretamente as mesmas funções usadas localmente
(`src.training.trainer.main`, `src.evaluation.evaluate.main`,
`src.optimization.export_onnx.main`) — a DAG só orquestra a ordem de
execução, sem duplicar lógica. Ela é criada **pausada**; dispare
manualmente pela UI ou `airflow dags trigger triage_training` (troque
`schedule=None` por, por exemplo, `"@weekly"` no arquivo da DAG para
retreino periódico automático).

## Otimização de Latência (Etapa 4)

A vetorização TF-IDF permanece em scikit-learn (já é barata); a etapa mais
cara — o classificador (Random Forest / Logistic Regression) — é exportada
para **ONNX** via `skl2onnx` e servida com `onnxruntime`
(`src/optimization/export_onnx.py`). `scripts/measure_latency.py` compara
os dois caminhos (`pipeline.predict` scikit-learn puro vs.
`vectorizer.transform` + sessão ONNX) e grava o comparativo em
`metrics/latency_comparison.json`:

```bash
make export-onnx
make benchmark-latency
cat metrics/latency_comparison.json
```

Para servir com o backend otimizado, defina `MODEL_BACKEND=onnx` no `.env`
(ou `MODEL_BACKEND=onnx docker compose up api`) antes de treinar/exportar.

## Dataset

Por padrão (`make dataset`), o repositório gera um **dataset sintético**
(`scripts/generate_synthetic_dataset.py`) de laudos em português, com as 3
classes balanceadas (`normal` / `atencao` / `urgente`, 3000 amostras por
padrão — configurável em `config/config.yaml` →
`data.n_synthetic_samples`). Isso existe só para que o pipeline seja
executável de ponta a ponta a partir de um clone limpo, sem depender de
credenciais externas.

Para usar um dataset real (recomendado antes da entrega final), qualquer
dataset com uma coluna de texto e uma coluna de rótulo serve — basta
substituir `data/raw/triage_reports.csv` mantendo as colunas `text`/`label`
(ou ajustar `data.text_column`/`data.label_column` em `config.yaml`).
Sugestões do enunciado: [Medical Abstracts TC
Corpus](https://www.kaggle.com/datasets/chaitanyakck/medical-text) (Kaggle)
ou recortes do [MIMIC-III](https://physionet.org/content/mimiciii/) (acesso
controlado).

## Testes

```bash
make lint
make test
make test-cov   # relatório HTML de cobertura em htmlcov/
```

Os testes não dependem de modelo treinado nem de dataset em disco — usam
dados sintéticos minúsculos gerados em memória, então passam em qualquer
clone limpo (é isso que o job `test` do CI valida).

## Critérios de Avaliação — onde cada um é atendido

| Critério | Peso | Onde |
|---|---|---|
| Modelagem e Otimização | 20% | `src/models/classifier.py`, `src/optimization/export_onnx.py`, `metrics/latency_comparison.json` |
| CI/CD (GitHub Actions) | 15% | `.github/workflows/ci.yml` |
| Orquestração (Airflow) | 15% | `airflow/dags/triage_training_dag.py`, `docker-compose.airflow.yml` |
| Monitoramento | 20% | `docker-compose.yml`, `src/serving/metrics.py`, `monitoring/` |
| Documentação (README) | 15% | este arquivo + `docs/architecture.md` |
| Vídeo STAR | 15% | link: _a adicionar após a gravação_ |

## Vídeo STAR

_Link: TODO — adicionar após a gravação (≤ 5 min, método STAR)._

## Troubleshooting

- **`poetry: command not found`** — use `python -m poetry` (é o que o
  `Makefile` já faz internamente) em vez de `poetry` direto, especialmente
  no Windows quando o script de entrada não está no `PATH`.
- **Porta ocupada (8000/9090/3000/8080)** — troque via `.env`
  (`API_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`, `AIRFLOW_PORT`).
- **`/health` retorna `degraded`** — nenhum modelo foi treinado ainda; rode
  `make train` (e `make export-onnx` se `MODEL_BACKEND=onnx`).
