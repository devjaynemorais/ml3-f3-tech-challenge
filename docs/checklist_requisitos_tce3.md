# De/para dos requisitos do Tech Challenge — Fase 3

Fonte de verdade: `docs/MLET - Tech Challenge Fase 3.pdf`. Este checklist liga
cada requisito obrigatório a uma evidência verificável no repositório. Ele não
substitui as instruções de execução do `README.md`.

| Requisito do enunciado | Evidência no projeto | Situação |
| --- | --- | --- |
| Classificador NLP funcional | LR + TF-IDF em `src/models/classifier.py`, artefato `models/artifacts/triage_pipeline.joblib` e métricas em `metrics/eval_metrics.json` | Coberto |
| API REST FastAPI | `src/serving/api.py`, endpoints `/`, `/health`, `/predict`, `/metrics` e `/docs` | Coberto |
| Dockerfile funcional para inferência | Stage `api` do `Dockerfile`; build local validado | Coberto |
| Decisão batch versus real-time e nuvem | README e `docs/architecture.md` | Coberto |
| Baseline local de latência | `scripts/measure_latency.py` e `metrics/latency_comparison.json` | Coberto |
| Otimização de performance | Conversão parcial do classificador em `src/optimization/export_onnx.py`; artefatos em `models/onnx/` | Coberto |
| Comparação original versus otimizado | README e JSON de benchmark: sklearn 4,14 ms, ONNX 3,37 ms, speedup 1,228x e redução de 18,5% | Coberto |
| CI/CD com lint, testes e build | `.github/workflows/ci.yml`, com jobs `quality` e `docker-build` em push e pull request | Coberto no código; execução remota depende de commit/push |
| Pelo menos duas automações de CI/CD | Lint Ruff, pytest e build Docker | Coberto |
| Pipeline de treino/re-treino em Airflow | `airflow/dags/triage_training_dag.py`: `ingest_data -> train_model -> evaluate_model -> export_onnx` | Coberto |
| Runtime do Airflow reproduzível | `Dockerfile.airflow` instala dependências `main,train` e recursos NLP usados pelo pipeline | Coberto |
| API instrumentada com contagem e duração | `src/serving/metrics.py` e middleware em `src/serving/api.py` | Coberto |
| Compose com API, Prometheus e Grafana | `docker-compose.yml` | Coberto |
| Dashboard Grafana com ao menos três painéis | `monitoring/grafana/dashboards/api_overview.json`: volume, latência, erros, status e distribuição por classe | Coberto |
| Print ou JSON do dashboard | JSON versionado em `monitoring/grafana/dashboards/api_overview.json` | Coberto |
| Dataset público com pelo menos 2.000 amostras | Medical Abstracts TC Corpus; origem, licença, 14.438 linhas e limitações no README e model card | Coberto |
| Experimentos e decisão do modelo | `docs/metodologia_experimentos.md`, com métricas, duração, inferência disponível, análise multirrótulo e critério de produção | Coberto |
| Histórico de commits semântico | Histórico Git com mensagens convencionais; padrão documentado no README | Coberto no histórico local |
| Instruções claras de execução | README: instalação, treino, API, Docker, Airflow, monitoramento e benchmark | Coberto |
| Vídeo STAR de até cinco minutos | Link e roteiro STAR devem ser preenchidos após a gravação | **Pendente — única exceção declarada** |

## Conferência do conteúdo do vídeo

Quando o vídeo for gravado, ele deve demonstrar em até cinco minutos:

1. **Situation**: problema de classificação/triagem e necessidade de resposta
   rápida, sem apresentar o sistema como diagnóstico clínico.
2. **Task**: API, latência, CI/CD, Airflow e monitoramento pedidos na fase.
3. **Action**: arquitetura real-time, LR + TF-IDF, conversão ONNX, workflow,
   DAG e dashboard Grafana.
4. **Result**: execução do pipeline, API respondendo, gráficos recebendo
   requisições, comparação de latência e principais limitações aprendidas.
