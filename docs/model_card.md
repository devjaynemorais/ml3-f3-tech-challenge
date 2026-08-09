# Model Card — Classificador de Triagem

> Preencha as seções marcadas com `TODO` depois de treinar com o dataset
> real que o grupo escolher (ver README, seção Dataset). Como está, este
> documento descreve o comportamento esperado com o dataset sintético
> padrão do repositório.

## Tarefa

Classificação de texto multiclasse: dado o texto de um laudo/relato
clínico, prever a urgência em `normal` / `atencao` / `urgente`.

## Dados

- **Padrão do repositório**: dataset sintético gerado por
  `scripts/generate_synthetic_dataset.py` — frases template em português
  combinadas com sintomas característicos de cada classe. Serve para
  validar o pipeline de ponta a ponta, **não** para avaliar qualidade real
  do classificador.
- **TODO**: substituir por um dataset real (Medical Abstracts TC Corpus,
  MIMIC-III ou equivalente, ≥ 2000 amostras) e re-treinar antes de reportar
  métricas para a entrega final.

## Modelo

- Pipeline scikit-learn: `TfidfVectorizer` (`config.yaml` →
  `features.max_features`, `ngram_range`, `min_df`) seguido de um
  classificador leve — `RandomForestClassifier` ou `LogisticRegression`
  (`config.yaml` → `model.type`).
- Pré-processamento de texto: minúsculas, remoção de acentos, normalização
  de espaços (`src/features/text_preprocessing.py`).

## Métricas

Geradas por `make evaluate` em `metrics/eval_metrics.json`: accuracy, F1
macro e `classification_report` completo (precision/recall/F1 por classe).

`TODO`: colar aqui os números obtidos com o dataset real escolhido.

## Otimização de latência

`metrics/latency_comparison.json` (gerado por `make benchmark-latency`)
compara a latência média/p50/p95/p99 do `pipeline.predict` scikit-learn
puro contra o classificador exportado para ONNX Runtime.

`TODO`: colar aqui o speedup observado.

## Limitações conhecidas

- O dataset sintético padrão usa um vocabulário de templates limitado —
  qualquer métrica obtida com ele **superestima** a qualidade real do
  modelo (o problema é artificialmente fácil).
- O modelo não usa nenhum contexto além do texto do laudo (sem sinais
  vitais estruturados, histórico do paciente, etc.) — é um classificador de
  triagem auxiliar, não um substituto de avaliação clínica.
- Três classes fixas (`normal`/`atencao`/`urgente`) — não captura
  gradações dentro de uma mesma classe nem contempla out-of-scope (ex.: um
  texto que não é um laudo médico).
