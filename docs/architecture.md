# Arquitetura

## Visão geral do fluxo

```
                    ┌─────────────────────┐
   laudo (texto) ──▶│   FastAPI  /predict  │──▶ label + scores + backend
                    │  (TF-IDF + RF/LR ou  │
                    │   TF-IDF + ONNX RT)  │
                    └──────────┬───────────┘
                               │ /metrics (prometheus_client)
                               ▼
                    ┌─────────────────────┐      ┌──────────────┐
                    │      Prometheus      │─────▶│   Grafana    │
                    └─────────────────────┘      └──────────────┘

  data/raw/*.csv ──▶ Airflow DAG (ingest → train → evaluate → export ONNX) ──▶ models/{artifacts,onnx}/
```

A API e o pipeline de treino compartilham o mesmo código em `src/` — a DAG
do Airflow e o `Makefile` só chamam as mesmas funções Python usadas
localmente, então "treinar local" e "retreinar via Airflow" nunca podem
divergir silenciosamente.

## Por que real-time (e não batch)

O objetivo é reduzir o tempo entre "laudo chegou" e "caso urgente foi
visto". Qualquer atraso de lote (mesmo que só de minutos) reintroduz o
problema que a triagem automática deveria resolver. Isso implica:

- A API precisa estar sempre no ar (não é um job que roda e desliga).
- Latência de inferência importa por requisição — daí a otimização ONNX
  (Etapa 4), não só um "nice to have".
- Escalar por carga de requisições (ex.: horários de pico de admissão) é
  mais natural com um serviço containerizado atrás de um load balancer do
  que com um pipeline batch agendado.

## Por que separar treino de serving no Docker

O `Dockerfile` tem stages `builder → train → api`. `train` inclui
`scripts/` (para a geração/ingestão de dados) e é a imagem usada tanto pelo
`docker compose run --rm train` quanto — implicitamente — pela lógica que a
DAG do Airflow chama (rodando fora do container `train` propriamente dito,
mas reaproveitando exatamente o mesmo `src/`). `api` só carrega o pipeline
já treinado (via `models/artifacts/` ou `models/onnx/`, montados como
volume) — não tem `scikit-learn` de treino nem `scripts/` embutidos, o que
mantém a imagem final menor e reduz a superfície de coisas que podem
quebrar em produção (o container que serve tráfego real não tem por que
saber gerar dataset sintético, por exemplo).

## Por que TF-IDF fica fora do grafo ONNX

`skl2onnx` converte bem estimadores como `RandomForestClassifier` e
`LogisticRegression`, mas converter um `TfidfVectorizer` inteiro para ONNX
exige runtime de tokenização adicional (`onnxruntime-extensions`) — custo de
complexidade desproporcional ao ganho, já que a vetorização TF-IDF em si já
é rápida (é uma multiplicação esparsa, não uma floresta com centenas de
árvores). Por isso a otimização mira o que de fato domina a latência: o
classificador. Ver `src/optimization/export_onnx.py` e a discussão em
`scripts/measure_latency.py`.

## Monitoramento: que métricas e por quê

- `http_requests_total{method,path,status_code}` — contagem de requisições;
  base para o painel de "total de requisições" e para calcular taxa de erro
  (`status_code=~"5.."`).
- `http_request_duration_seconds{method,path}` (histograma) — base para
  `histogram_quantile` no painel de latência p50/p95; é a métrica mais
  próxima do que efetivamente afeta um caso urgente sendo visto a tempo.
- `triage_predictions_total{predicted_label}` — não é uma métrica de
  infraestrutura, é uma métrica de negócio/domínio: acompanha o volume de
  cada classe de urgência ao longo do tempo, o que ajuda a notar deriva de
  distribuição (ex.: um surto sazonal fazendo `urgente` disparar) —
  informação relevante tanto para capacidade quanto para monitorar se o
  modelo continua se comportando como esperado.

## Deploy em nuvem — ver README

A decisão completa (AWS ECS Fargate + ECR + ALB, alternativas Azure/GCP
consideradas) está na seção "Decisão Arquitetural" do `README.md`, conforme
pedido no enunciado (Etapa 1).
