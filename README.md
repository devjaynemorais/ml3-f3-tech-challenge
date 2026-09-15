# Medical Text Classifier API

Sistema MLOps de classificação multilabel de abstracts médicos públicos em
inglês, desenvolvido para o Tech Challenge da Fase 3 de Machine Learning
Engineering da FIAP. O pipeline prepara o Medical Abstracts TC Corpus, treina
um classificador NLP leve e o serve por uma API FastAPI observável.

O sistema é uma demonstração técnica de apoio à categorização e priorização.
Ele não realiza diagnóstico, não representa triagem hospitalar real e não
substitui revisão humana ou validação clínica.

## Vídeo STAR da apresentação

A apresentação do projeto no formato STAR está disponível no YouTube:

[Assistir ao vídeo STAR](https://youtu.be/otX_U_mz_kk)

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
- features por TF-IDF (padrão) ou embeddings biomédicos pré-treinados (spaCy
  `en_core_sci_md`), selecionável no YAML;
- cinco classificadores binários One-vs-Rest com Regressão Logística
  (padrão — melhor F1-macro em
  validação cruzada entre os comparados), Complement Naive Bayes, Linear SVM
  calibrado, Random Forest ou Gradient Boosting, todos selecionáveis no YAML;
- busca de hiperparâmetros por validação cruzada (`tuning.enabled` no YAML) —
  `make train` treina com CV sobre treino+validação e persiste o vencedor já
  re-ajustado nos dois splits;
- pipeline sklearn persistido e classificador exportável para ONNX Runtime;
- API FastAPI com modo degradado quando não há artefato válido;
- Prometheus e Grafana para volume, erros, latência e distribuição de classes;
- Airflow para `ingest_data -> train_model -> evaluate_model -> export_onnx`;
- GitHub Actions para lint, testes, teste arquitetural e build da imagem.

Detalhes e decisões estão em [docs/architecture.md](docs/architecture.md) e as
limitações do modelo em [docs/model_card.md](docs/model_card.md). O
diagnóstico completo do teto de F1 do corpus e as propostas de melhoria
avaliadas estão em [docs/plano_melhoria_f1.md](docs/plano_melhoria_f1.md).
A comparação reproduzível dos quatro experimentos oficiais está em
[docs/metodologia_experimentos.md](docs/metodologia_experimentos.md). O de/para
completo do enunciado está em
[docs/checklist_requisitos_tce3.md](docs/checklist_requisitos_tce3.md).

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

O projeto usa a versão processada do **Medical Abstracts Text Classification
Corpus**, publicada por Schopf, Braun e Matthes. A fonte canônica é o
[repositório dos autores](https://github.com/sebischair/Medical-Abstracts-TC-Corpus),
também espelhado no
[Kaggle](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus).
Essa versão contém 14.438 linhas rotuladas em inglês: 11.550 no treino oficial
e 2.888 no teste oficial, distribuídas em cinco categorias. O corpus processado
é disponibilizado sob **Creative Commons CC BY-SA 3.0**; redistribuições e
trabalhos derivados devem preservar atribuição e compartilhamento pela mesma
licença. A referência acadêmica é Schopf, Braun e Matthes (NLPIR 2022/ACM 2023),
DOI [`10.1145/3582768.3582795`](https://doi.org/10.1145/3582768.3582795).

Use os três arquivos:

```text
data/raw/medical_tc_train.csv
data/raw/medical_tc_test.csv
data/raw/medical_tc_labels.csv
```

Os arquivos de treino e teste devem conter `condition_label` e
`medical_abstract`; o arquivo de labels deve conter `condition_label` e
`condition_name`. `make dataset` processa treino e teste separadamente, agrega
os labels repetidos de cada abstract, remove somente do treino qualquer texto
presente no teste e retira 10% do treino seguro para validação. Os arquivos
processados têm uma coluna de texto e cinco targets binários:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

O teste preserva seus 2.770 abstracts únicos e nunca participa do fit. Foram
encontrados 988 textos nos dois CSVs oficiais; todas as 1.097 linhas desses
textos são descartadas somente do treino. Nenhum label é transferido entre
splits. O resultado é treino+validação com 8.457 abstracts e sobreposição zero.

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

`make install` instala as dependências, as stopwords NLTK, o modelo spaCy
`en_core_web_sm 3.8.0` (preprocessing) e o `en_core_sci_md 0.5.4` do scispaCy
(embeddings biomédicos, usado só quando `features.type: embeddings`). No
Windows, quando `make` não estiver disponível, use o interpretador da `.venv`
com os módulos indicados nos targets do Makefile.

Para reproduzir o experimento BERT, instale também o grupo opcional pesado e
execute o target dedicado:

```bash
make install-experiments
make finetune-bert
```

`TRAINING_DEVICE` no `.env` controla o dispositivo do fine-tuning do BERT
(`auto` detecta GPU se disponível; `cpu`/`cuda` forçam um dos dois — `cuda`
falha com erro claro se nenhuma GPU for detectada).

Essa mesma variável também acelera `model.type: gradient_boosting` em
`make train`/`make experiments`: com `TRAINING_DEVICE=cuda` (ou `gpu`)
**explícito**, o `GradientBoostingStrategy` troca o `GradientBoostingClassifier`
do scikit-learn (sem backend de GPU) por XGBoost com `device="cuda"`. Só ativa
com pedido explícito — nunca por auto-detecção, mesmo numa máquina com GPU —
porque é uma troca de algoritmo, não só de velocidade; os hiperparâmetros
tunados de `config.yaml` foram validados contra o sklearn, não o XGBoost.
Exportação para ONNX não é suportada para modelos treinados assim (sirva via
`MODEL_BACKEND=sklearn` ou retreine em CPU para exportar). Os demais
`model.type` e TF-IDF/embeddings não têm equivalente de GPU nativo no Windows
e ignoram a variável, só avisando no log se `cuda`/`gpu` for pedido.

Depois de iniciar a API, abra `http://localhost:8000/docs`.

## API

| Método | Rota | Contrato |
|---|---|---|
| GET | `/` | nome e versão do serviço |
| GET | `/health` | `ok` com backend ou `degraded` sem artefato válido |
| POST | `/predict` | recebe `text` e retorna `labels`, `label` principal, cinco scores independentes e `backend` |
| GET | `/metrics` | métricas Prometheus |
| GET | `/docs` | Swagger UI |
| GET | `/explain?text=...` | mesma coisa, mas com o passo a passo (pré-processamento real, termos TF-IDF que mais pesaram) — usado pela demo abaixo |
| GET | `/demo` | página HTML da demo interativa |
| GET | `/demo/sample-texts` | 5 abstracts reais de exemplo, um por categoria canônica |

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

### Demo interativa

Com um modelo treinado (`make train`), rode `make demo` (atalho para `make api`
que já imprime a URL) e abra **`http://localhost:8000/demo`**: uma
página que chama o mesmo `TriagePredictor` Production de verdade (via
`/explain`, a mesma lógica de `/predict` com o passo a passo exposto —
pré-processamento real do texto, termos TF-IDF que mais pesaram para a
categoria prevista, ranking entre as 5 categorias) e narra as 6 etapas do
`Makefile` que rodaram offline antes disso. Serve tanto pra mostrar o projeto
funcionando quanto de roteiro visual pro vídeo STAR.

## Docker e observabilidade

```bash
make compose-build
make compose-up
```

Serviços locais:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin`/`admin` por padrão)
- MLflow: `http://localhost:5000`

O dashboard provisionado apresenta total e taxa de requisições, taxa de erro,
latência p50/p95 e distribuição das categorias médicas. As métricas preservadas
são `http_requests_total`, `http_request_duration_seconds` e
`triage_predictions_total`.

## MLflow

Cada `make train` registra no MLflow os hiperparâmetros do modelo
selecionado, `model_type`, `n_train_samples` e o pipeline treinado como
artefato; o `make evaluate` seguinte reabre a mesma run (via `run_id`
salvo em `model_metadata.json`) e anexa accuracy, macro/weighted F1 e
recall médio das classes minoritárias de validação e teste. Runs nascem
nomeadas `{model_type}-tfidf`, para comparar diferentes Strategies lado a
lado. Se o servidor MLflow não estiver acessível, o treino/avaliação
continuam normalmente e só um aviso é logado — o tracking nunca bloqueia o
pipeline.

Duas formas de subir o servidor, escolha uma:

```bash
# Local (sem Docker) — inicia na hora, sem pull/build de imagem
make mlflow

# Ou containerizado, junto do resto da stack
docker compose up -d mlflow
```

Depois, treine normalmente:

```bash
make train
make evaluate
```

`MLFLOW_TRACKING_URI` no `.env` controla o endpoint usado localmente
(`http://localhost:5000` nos dois casos acima); dentro do compose, o
serviço `train` sobrescreve para `http://mlflow:5000`. O backend local
(`make mlflow`) usa `mlflow.db`/`mlartifacts/` na raiz do projeto; o
containerizado usa `mlflow-data/` — são históricos independentes.

### Model Registry

Depois de `make train` + `make evaluate`, `make promote` busca no experimento o
run com melhor `registry.metric` (padrão: `cv_macro_f1_mean`, configurável em
`config/config.yaml`) e desempata pelo menor `cv_macro_f1_std`. A seleção usa
CV estratificada de 3 folds sobre treino+validação e nunca usa o teste oficial,
evitando promover um modelo por desempenho observado no teste. O comando
registra o artefato no MLflow Model Registry e promove para `registry.stage`
(padrão `Production`). O resultado fica em
`models/promoted_model.json` (fora do Git) e na aba **Models** da UI do
MLflow.

```bash
make promote
```

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

No benchmark oficial local, com 5 warm-ups e 200 predições unitárias do mesmo
texto sintético, o pipeline sklearn levou **3,04 ms/texto** em média e o backend
ONNX **2,72 ms/texto**. Isso representa speedup de **1,118x** e redução média de
latência de **10,5%**. Os valores são dependentes do hardware e do modelo
promovido no momento; o comando abaixo reproduz a comparação no ambiente
corrente e grava os números atuais em `metrics/latency_comparison.json`.

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
