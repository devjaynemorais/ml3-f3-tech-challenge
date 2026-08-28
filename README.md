# Medical Text Classifier API

Sistema MLOps de classificação multiclasse de abstracts médicos públicos em
inglês, desenvolvido para o Tech Challenge da Fase 3 de Machine Learning
Engineering da FIAP. O pipeline prepara o Medical Abstracts TC Corpus, treina
um classificador NLP leve e o serve por uma API FastAPI observável.

O sistema é uma demonstração técnica de apoio à categorização e priorização.
Ele não realiza diagnóstico, não representa triagem hospitalar real e não
substitui revisão humana ou validação clínica.

## Categorias

A ordem canônica usada em configuração, artefatos e respostas da API é:

1. `neoplasms`
2. `digestive system diseases`
3. `nervous system diseases`
4. `cardiovascular diseases`
5. `general pathological conditions`

## Arquitetura

O serviço foi mantido real-time porque o objetivo técnico é disponibilizar uma
classificação por requisição com baixa latência. A arquitetura combina:

- preprocessing configurável por Strategy e Factory;
- TF-IDF com Regressão Logística por padrão;
- Random Forest e Gradient Boosting selecionáveis no YAML;
- pipeline sklearn persistido e classificador exportável para ONNX Runtime;
- API FastAPI com modo degradado quando não há artefato válido;
- Prometheus e Grafana para volume, erros, latência e distribuição de classes;
- Airflow para `ingest_data -> train_model -> evaluate_model -> export_onnx`;
- GitHub Actions para lint, testes, teste arquitetural e build da imagem.

Detalhes e decisões estão em [docs/architecture.md](docs/architecture.md) e as
limitações do modelo em [docs/model_card.md](docs/model_card.md).

## Decisão de deploy em nuvem

O desenho recomendado usa AWS ECR para imagens, ECS Fargate para executar a API
e Application Load Balancer com health check em `/health`. CloudWatch pode
receber métricas operacionais; a instrumentação Prometheus deste repositório
também pode ser conectada ao Amazon Managed Service for Prometheus. Para
retreino, a DAG pode migrar para Amazon MWAA.

Azure Container Apps/Azure ML Endpoints e GCP Cloud Run são alternativas
equivalentes. A recomendação por AWS considera familiaridade operacional e não
uma limitação técnica do classificador.

## Estrutura principal

```text
config/config.yaml                  configuração tipada do experimento
data/raw/                           três CSVs originais, fora do Git
data/processed/                     treino, validação e teste canônicos
src/data/                           validação, mapeamento e persistência
src/features/                       Strategies e Factory de preprocessing
src/models/                         Strategies, Factory e registro do pipeline
src/training/                       treino apenas no split processado de treino
src/evaluation/                     métricas de validação e teste oficial
src/optimization/                   exportação do classificador para ONNX
src/serving/                        app factory, rotas, schemas e predictors
airflow/dags/triage_training_dag.py orquestração manual do pipeline
monitoring/                         Prometheus e dashboard Grafana
postman/                            coleção e cinco exemplos JSON
tests/                              testes unitários, integração e arquitetura
```

## Dataset

Use os três arquivos do Medical Abstracts TC Corpus:

```text
data/raw/medical_tc_train.csv
data/raw/medical_tc_test.csv
data/raw/medical_tc_labels.csv
```

Os arquivos de treino e teste devem conter `condition_label` e
`medical_abstract`; o arquivo de labels deve conter `condition_label` e
`condition_name`. `make dataset` valida os schemas e o mapeamento, retira 10%
estratificados somente do treino oficial e persiste:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

O teste oficial permanece integral e nunca participa do fit. Dados, modelos e
métricas gerados ficam fora do Git. Verifique licença e termos da fonte antes de
redistribuir o corpus.

## Início rápido

```bash
make install
cp .env.example .env
make dataset
make train
make evaluate
make export-onnx
make benchmark-latency
make api
```

`make install` instala as dependências, as stopwords NLTK e o modelo spaCy
`en_core_web_sm 3.8.0`. No Windows, quando `make` não estiver disponível, use o
interpretador da `.venv` com os módulos indicados nos targets do Makefile.

Depois de iniciar a API, abra `http://localhost:8000/docs`.

## API

| Método | Rota | Contrato |
|---|---|---|
| GET | `/` | nome e versão do serviço |
| GET | `/health` | `ok` com backend ou `degraded` sem artefato válido |
| POST | `/predict` | recebe `text` e retorna `label`, cinco `scores` e `backend` |
| GET | `/metrics` | métricas Prometheus |
| GET | `/docs` | Swagger UI |

Exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Echocardiography showed reduced left ventricular function."}'
```

Texto vazio retorna HTTP 422. Sem modelo carregado, `/health` continua acessível
em modo degradado e `/predict` retorna HTTP 503. Textos recebidos não são
incluídos nos logs.

A coleção [postman/Triage-API.postman_collection.json](postman/Triage-API.postman_collection.json)
contém todos os GETs e cinco exemplos sintéticos de predição.

## Docker e observabilidade

```bash
make compose-build
make compose-up
```

Serviços locais:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin`/`admin` por padrão)

O dashboard provisionado apresenta total e taxa de requisições, taxa de erro,
latência p50/p95 e distribuição das categorias médicas. As métricas preservadas
são `http_requests_total`, `http_request_duration_seconds` e
`triage_predictions_total`.

## Airflow

```bash
make airflow-up
# UI: http://localhost:8080 (admin/admin)
make airflow-down
```

A imagem definida em `Dockerfile.airflow` instala dependências e recursos NLP
durante o build. A DAG `triage_training` nasce pausada (`schedule=None`) e
reutiliza as funções de `src/`, sem duplicar lógica de ML.

## ONNX e latência

O artefato `models/onnx/vectorizer.joblib` contém todo o prefixo anterior ao
classificador: preprocessing, TF-IDF e, quando aplicável, seleção/conversão de
features. Apenas o classificador final é convertido para ONNX. O benchmark usa
o mesmo contrato `TriagePredictor` nos dois backends e grava
`metrics/latency_comparison.json`.

Defina `MODEL_BACKEND=onnx` no `.env` para servir o backend otimizado depois de
executar `make export-onnx`.

## Qualidade e CI/CD

```bash
make lint
make test
make test-cov
```

Os testes usam fixtures mínimas em memória e não dependem dos CSVs locais. O
teste arquitetural analisa a AST e rejeita funções de `src/` com mais de 20
linhas lógicas. Ruff mantém complexidade McCabe máxima 10.

O workflow do GitHub Actions executa lint, testes com cobertura e build do
stage Docker `api` em pushes e pull requests para `main`.

## Critérios da entrega

| Critério | Evidência |
|---|---|
| Modelagem e otimização | `src/models/`, `src/optimization/`, métricas geradas |
| CI/CD | `.github/workflows/ci.yml` |
| Orquestração | DAG Airflow e `Dockerfile.airflow` |
| Monitoramento | `docker-compose.yml`, Prometheus e Grafana |
| Documentação | este README, arquitetura e model card |
| Vídeo STAR | _link a adicionar somente após a gravação_ |

## Troubleshooting

- `poetry: command not found`: use `python -m poetry` ou o executável Python da
  `.venv`.
- erro de stopwords ou `en_core_web_sm`: execute `make nlp-resources`.
- `/health` degradado: execute treino e, para ONNX, também a exportação; confira
  `MODEL_BACKEND` e os paths em `.env`.
- portas ocupadas: ajuste `API_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT` ou
  `AIRFLOW_PORT` no `.env`.
