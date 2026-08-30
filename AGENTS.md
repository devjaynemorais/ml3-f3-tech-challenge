# AGENTS.md

Guia operacional para agentes trabalhando neste repositório.

## Objetivo do projeto

O objetivo é entregar um sistema de classificação de abstracts médicos públicos
em inglês, com um classificador NLP leve servido por API REST em container
Docker.

O projeto deve evidenciar, de ponta a ponta:

- API FastAPI para inferência em tempo real.
- Pipeline CI/CD no GitHub Actions com lint, testes e build.
- Pipeline simples de treino/retreino orquestrado por Airflow.
- Stack local de monitoramento com API, Prometheus e Grafana via Docker Compose.
- Otimização de latência, atualmente via exportação parcial para ONNX Runtime.
- Documentação clara da decisão arquitetural e instruções de execução.
- Histórico de commits semântico e organizado.

O modelo classifica textos em `neoplasms`, `digestive system diseases`,
`nervous system diseases`, `cardiovascular diseases` e
`general pathological conditions`. Ele é um apoio à categorização e
priorização, não um diagnóstico clínico nem substituto de revisão humana.

## Mapa rápido do repositório

- `README.md`: visão geral, decisão de deploy em nuvem, instruções de execução,
  critérios de avaliação e espaço para o link do vídeo STAR.
- `docs/MLET - Tech Challenge Fase 3.pdf`: enunciado oficial do desafio.
- `docs/architecture.md`: detalhes da arquitetura real-time, Docker,
  observabilidade e decisão de deploy.
- `docs/model_card.md`: ficha do modelo, métricas, limitações e TODOs para
  dataset real.
- `docs/specs/spec_notebook_eda_nlp_hospital.md`: especificação do notebook
  exploratório, separada do pipeline de produção.
- `notebooks/01_eda_nlp_hospital.ipynb`: EDA autocontida, sem modelagem.
- `notebooks/02_baseline_modelagem.ipynb`: modelagem e avaliação de baselines,
  autocontido, consumindo os achados do notebook 01.
- `config/config.yaml`: configuração de experimento, dados, labels, split,
  features, modelo e nomes de artefatos.
- `.env.example`: configuração de runtime da API e dos serviços locais.
- `src/data/`: leitura, validação e split dos dados.
- `src/features/`: preprocessamento textual usado pelo pipeline sklearn.
- `src/models/`: construção e persistência do pipeline de classificação.
- `src/training/`: treino do modelo.
- `src/evaluation/`: avaliação e escrita de métricas em `metrics/`.
- `src/optimization/`: exportação do classificador para ONNX.
- `src/serving/`: API, schemas, carregamento do modelo e métricas Prometheus.
- `scripts/`: benchmark de latência com textos sintéticos não sensíveis.
- `airflow/dags/triage_training_dag.py`: DAG de ingestão, treino, avaliação e
  exportação ONNX.
- `monitoring/`: Prometheus, provisioning do Grafana e dashboard versionado.
- `tests/`: testes unitários e de serving.
- `Dockerfile`: stages `builder`, `train` e `api`.
- `docker-compose.yml`: API, Prometheus e Grafana.
- `docker-compose.airflow.yml`: Airflow local com Postgres.

## Contratos que devem ser preservados

### Configuração

- Mantenha hiperparâmetros, paths de dados, labels, split, features, tipo de
  modelo e nomes de artefatos em `config/config.yaml`.
- Mantenha configuração de runtime em `.env` ou `.env.example`: paths base dos
  artefatos, `MODEL_BACKEND`, portas e senhas locais.
- Não mova hiperparâmetros de treino para variáveis de ambiente sem motivo
  arquitetural claro.

### Dados

- Os CSVs brutos são `data/raw/medical_tc_train.csv`,
  `data/raw/medical_tc_test.csv` e `data/raw/medical_tc_labels.csv`.
- Treino/teste esperam `condition_label` e `medical_abstract`; o mapeamento
  espera `condition_label` e `condition_name`.
- As labels válidas, na ordem canônica, são `neoplasms`,
  `digestive system diseases`, `nervous system diseases`,
  `cardiovascular diseases` e `general pathological conditions`.
- A validação usa 10% estratificados somente do treino oficial; o teste oficial
  deve permanecer integral e fora do fit.
- Para a entrega final, documente origem, tamanho,
  colunas, licença, limitações e impactos nas métricas.
- Em dados clínicos ou hospitalares, não exponha identificadores de pacientes,
  texto sensível ou exemplos que possam violar privacidade.

### Modelo e artefatos

- O pipeline base concatena as Strategies configuradas de preprocessamento,
  `TfidfVectorizer` e uma Strategy de `LogisticRegression`,
  `RandomForestClassifier` ou `GradientBoostingClassifier`, construído em
  `src/models/classifier.py`.
- O pipeline sklearn treinado deve ser salvo em
  `models/artifacts/triage_pipeline.joblib`.
- A metadata deve ser salva em `models/artifacts/model_metadata.json`.
- A exportação ONNX salva:
  - `models/onnx/triage_classifier.onnx`;
  - `models/onnx/vectorizer.joblib`;
  - `models/onnx/classes.json`.
- Todo o prefixo de features permanece em sklearn no backend ONNX:
  preprocessamento, TF-IDF e eventuais seleção/conversão. Apenas o classificador
  final é convertido. Não altere isso sem atualizar
  `src/optimization/export_onnx.py`, `src/serving/model_loader.py`, benchmark,
  docs e testes.
- Qualquer novo backend de inferência deve implementar o contrato
  `TriagePredictor`: `backend` e `predict(text) -> (label, scores)`.

### API

Endpoints públicos esperados:

- `GET /`: metadados do serviço.
- `GET /health`: estado de carregamento do modelo.
- `POST /predict`: recebe `{"text": "..."}` e retorna `label`, `scores` e
  `backend`.
- `GET /metrics`: métricas Prometheus.
- `GET /docs`: Swagger UI gerado pelo FastAPI.

Preserve o comportamento degradado:

- Sem modelo carregado, `/health` deve retornar `{"status": "degraded",
  "model_loaded": false}`.
- Sem modelo carregado, `/predict` deve retornar HTTP 503.
- A API pode subir sem artefato treinado para permitir healthcheck de
  infraestrutura e diagnóstico claro.

### Observabilidade

As métricas Prometheus esperadas são:

- `http_requests_total{method,path,status_code}`;
- `http_request_duration_seconds{method,path}`;
- `triage_predictions_total{predicted_label}`.

O dashboard do Grafana deve manter pelo menos três visões alinhadas ao
enunciado: volume de requisições, latência/tempo de resposta e taxa de erro.
Distribuição de predições por classe é uma métrica de domínio útil e deve ser
preservada quando possível.

### Airflow

- A DAG principal é `triage_training`.
- O fluxo esperado é `ingest_data -> train_model -> evaluate_model ->
  export_onnx`.
- A DAG deve reutilizar funções de `src/`; não duplique lógica de treino dentro
  do arquivo da DAG.
- A DAG nasce pausada (`schedule=None`) e pode ser acionada manualmente para
  demonstração.

### Docker

- Preserve os stages do `Dockerfile`:
  - `builder`: instala dependências principais com Poetry;
  - `train`: roda jobs de treino, avaliação e otimização;
  - `api`: serve a FastAPI.
- A imagem `api` deve ser focada em serving.
- `docker-compose.yml` deve subir API, Prometheus e Grafana.
- `docker-compose.airflow.yml` deve ficar separado da stack principal para
  manter o ambiente local mais simples.

## Comandos de trabalho

Use os comandos do `Makefile` como interface principal:

```bash
make install
make lint
make test
make test-cov
make dataset
make train
make evaluate
make export-onnx
make benchmark-latency
make pipeline
make api
make compose-build
make compose-up
make compose-down
make airflow-up
make airflow-down
```

No Windows, se `poetry` não estiver no `PATH`, prefira o padrão já usado pelo
`Makefile`: `python -m poetry`.

## Validação antes de finalizar mudanças

Escolha a validação proporcional ao tipo de mudança:

- Documentação apenas: revise links, paths, comandos e aderência ao enunciado.
- Código Python: rode `make lint` e `make test`.
- API ou serving: rode testes e verifique `/health`, `/predict`, `/metrics` e
  `/docs`.
- Pipeline de treino ou dados: rode `make dataset`, `make train`,
  `make evaluate` ou `make pipeline`, conforme o impacto.
- ONNX ou latência: rode `make export-onnx` e `make benchmark-latency`.
- Docker: rode `make compose-build`; quando possível, suba a stack com
  `make compose-up`.
- Airflow: suba com `make airflow-up` e confirme que a DAG `triage_training`
  aparece e executa as tasks na ordem correta.

Se não for possível rodar uma validação esperada, registre claramente o motivo
no resumo final ou na documentação alterada.

## Regras de desenvolvimento

- Leia os arquivos atuais antes de editar. O repositório pode ter mudanças não
  commitadas.
- Não reverta alterações de terceiros sem pedido explícito.
- Prefira alterações pequenas, coerentes com a arquitetura existente.
- Não introduza frameworks novos se o stack atual resolve o requisito.
- Preserve Python 3.11, Poetry, Ruff e pytest como base de desenvolvimento.
- Ao tocar em contratos públicos, atualize testes e documentação juntos.
- Ao alterar modelos, dados, métricas ou thresholds, atualize `docs/model_card.md`
  e os artefatos em `metrics/` quando aplicável.
- Não versionar dados sensíveis, modelos grandes ou segredos. Use `.env` local
  para valores privados e mantenha `.env.example` sem credenciais reais.

## Commits e entrega

O enunciado exige histórico semântico e organizado. Use mensagens curtas no
formato convencional, por exemplo:

```text
Adicionar monitoramento da API de triagem

feat(monitoring): incluir métricas Prometheus e dashboard Grafana
```

Tipos comuns neste projeto: `feat`, `fix`, `docs`, `test`, `build`, `perf`,
`style`, `refactor`, `chore`, `ci`, `cleanup` e `remove`.

Antes de considerar a entrega pronta, confira a rubrica:

- Modelagem e otimização: modelo funcional e comparação de latência registrada.
- CI/CD: workflow do GitHub Actions com lint/test/build.
- Orquestração: DAG Airflow funcional.
- Monitoramento: Compose com API, Prometheus, Grafana e dashboard.
- Documentação: README com decisão de arquitetura em nuvem e execução clara.
- Vídeo STAR: link real adicionado somente depois da gravação.

## Limites clínicos e comunicação

- Nunca apresente o sistema como diagnóstico automatizado.
- Descreva o modelo como apoio de priorização ou triagem.
- Declare que o Medical Abstracts TC Corpus contém abstracts públicos em inglês
  e não representa triagem hospitalar real nem validação clínica.
- Para claims clínicos, exija dataset real, validação adequada, revisão humana,
  controle de falsos negativos, rastreabilidade e avaliação ética.
