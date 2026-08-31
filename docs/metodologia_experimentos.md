# Metodologia dos Experimentos

> **Status:** retrato histórico — os números e run IDs abaixo (`in_set_accuracy`,
> `accuracy_by_label_count`, `top_2_accuracy`) foram medidos **antes** da
> migração P3 para multirrótulo genuíno (`OneVsRestClassifier` sobre labels
> agregados por abstract, `src/data/make_dataset.py::aggregate_multilabel_split`;
> ver `docs/model_card.md`). Essas métricas reconstruíam o conjunto de rótulos
> válidos por abstract a partir de linhas duplicadas para compensar um modelo
> single-label — hoje esse conjunto já é o próprio target multi-hot da linha,
> então **as métricas justas atuais são `macro_avg.f1` (F1-macro multirrótulo,
> exibida como `test_f1_macro`) e `jaccard_samples` (accuracy multirrótulo por
> amostra, Godbole & Sarawagi 2004, exibida como `ml_accuracy`)** — priorize
> essas duas. Ver `docs/model_card.md#avaliação` para a definição corrente e
> `metrics/eval_metrics.json`/`metrics/experiment_comparison.json` para os
> números vigentes (gerados por `make experiments`).

## Objetivo

Este documento registra os 4 experimentos oficiais escolhidos para demonstrar a
evolução metodológica do projeto (apresentação/vídeo), entre os 7 candidatos
efetivamente testados durante a investigação de melhoria de F1 descrita em
`docs/plano_melhoria_f1.md`. Os números aqui são um retrato fixo, obtidos e
validados nesta sessão (single-label, ver banner de status acima); para o
estado corrente do artefato servido, ver `metrics/eval_metrics.json` e
`docs/model_card.md`.

## Metodologia comum a todos os experimentos

- **Split**: 90% treino / 10% validação (estratificado, `random_state=42`)
  sobre `medical_tc_train.csv`; teste oficial (`medical_tc_test.csv`) mantido
  íntegro e avaliado só ao final.
- **Seleção de hiperparâmetros**: busca em grade com `StratifiedKFold` (3
  folds) sobre **treino + validação combinados**, otimizando `f1_macro`. O
  pipeline vencedor é reajustado (refit) em todo esse conjunto — por isso a
  seção `validation` de `eval_metrics.json` vira um resumo de estatísticas de
  CV (`cv_macro_f1_mean`/`std`), não uma reavaliação enviesada.
- **Critério de promoção**: `cv_macro_f1_mean`, com desempate por
  `cv_macro_f1_std` (menor variância entre folds), em vez de comparar por uma
  métrica de split único — com ~1.150 linhas de validação, diferenças abaixo
  de ~0.015 são ruído estatístico, não sinal (ver `plano_melhoria_f1.md`,
  seção 3.6, para o caso real em que isso invalidou uma comparação anterior).
- **Métricas honestas**: `in_set_accuracy`, `accuracy_by_label_count` e
  `top_2_accuracy` contextualizam o F1-macro diante de um corpus multi-rótulo
  achatado em multiclasse (ver seção seguinte e `plano_melhoria_f1.md`).
- **Teto teórico**: o corpus tem abstracts repetidos com rótulos diferentes
  igualmente válidos (26% dos abstracts). Um oráculo que sorteia entre os
  rótulos válidos de cada abstract atinge F1-macro ≈ 0.7764 no teste oficial
  (`scripts/analyze_corpus.py`) — este é o teto real de qualquer classificador
  single-label neste corpus, não um limite do modelo ou das features.

## Os 4 experimentos oficiais

| # | Experimento | CV F1-macro (média ± desvio) | Test F1-macro | Test accuracy | Test recall minorias | Duração registrada | Inferência média | MLflow run |
| - | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Logistic Regression + TF-IDF** (`C=0.3`, produção) | **0.6165 ± 0.0023** | **0.6169** | 0.6153 | **0.735** | 2 min 09 s | **4,14 ms/texto** | `19c89237` |
| 2 | Gradient Boosting + TF-IDF | 0.5459 ± 0.0090 | 0.5268 | 0.5526 | 0.315 | 3 min 00 s | não medida | `4ded3f56` |
| 3 | Logistic Regression + embeddings biomédicos (`en_core_sci_md`) | 0.5959 ± 0.0039 | 0.6049 | 0.6066 | 0.731 | 4 min 43 s | não medida | `08771ad2` |
| 4 | BERT genérico fine-tunado (`bert-base-uncased`, 3 épocas, sem CV) | — | **0.6311** | **0.6340** | — (precision=0.622 / recall=0.654) | 12 min 00 s (GPU) | 6,05 ms/texto¹ | `4fd2bfc2` |

Notas:

- Nos experimentos 1–3, o tempo corresponde à duração total da run no MLflow
  (treino com CV/refit, avaliação e persistência), arredondada ao segundo. Para
  o BERT, o tempo é o `train_runtime` de 719,63 s registrado pelo Hugging Face
  Trainer na nova run; a duração externa da run no MLflow não representa o
  treino porque ela é aberta somente para registrar os resultados finais.
- A latência do experimento 1 vem de `metrics/latency_comparison.json`: média de
  200 predições unitárias com o pipeline sklearn, após 5 warm-ups, no ambiente
  local. Os experimentos 2 e 3 não tiveram benchmark de inferência persistido.
  ¹No BERT, 6,05 ms/texto é derivado do throughput de teste registrado pelo
  Hugging Face Trainer (165,242 amostras/s, batch 32); é processamento batelado
  em GPU, não latência online unitária, e não deve ser comparado diretamente
  com os 4,14 ms do serving sklearn.
- O experimento 4 (BERT) não passou pelo mesmo processo de CV — fine-tuning é
  caro demais (~12 min em GPU por rodada) para 3 folds. O número de teste é um
  ponto único, não uma média com desvio-padrão como os outros três; é o mais
  forte em F1/accuracy de teste, mas com menos evidência estatística de que
  essa vantagem se sustenta fora dessa amostra.
- O experimento 2 (Gradient Boosting) usa os hiperparâmetros default de
  `config.yaml` — não entrou na grade de tuning porque testes preliminares já
  mostravam desempenho bem abaixo dos modelos lineares em TF-IDF esparso de
  alta dimensão (consistente com a literatura: boosting não é a escolha
  natural para esse tipo de feature).
- O experimento 3 (embeddings) troca a matriz esparsa TF-IDF por vetores
  médios de palavras treinados em texto biomédico — captura similaridade
  semântica (ex.: "carcinoma" ~ "tumor") que o TF-IDF não vê, mas fica abaixo
  do TF-IDF no critério de promoção (CV F1-macro).

## Comparação consciente da estrutura multirrótulo

Embora os modelos atuais retornem uma única classe, o corpus contém abstracts
repetidos sob mais de um rótulo igualmente válido. A `in_set_accuracy` conta
como acerto qualquer previsão pertencente ao conjunto completo de rótulos
válidos do abstract; a `top_2_accuracy` verifica se o rótulo da linha aparece
entre as duas maiores probabilidades previstas. Elas contextualizam a accuracy
tradicional, mas não transformam o pipeline em um classificador multirrótulo.

| Experimento | Test accuracy | `in_set_accuracy` | `top_2_accuracy` |
| --- | --- | --- | --- |
| **Logistic Regression + TF-IDF** (`C=0.3`, produção) | 0.6153 | **0.8476** | **0.8927** |
| Gradient Boosting + TF-IDF | 0.5526 | 0.7957 | 0.8778 |
| Logistic Regression + embeddings biomédicos | 0.6066 | 0.8161 | 0.8816 |
| BERT genérico fine-tunado | **0.6340** | **0.8944** | **0.9107** |

Na comparação completa, o BERT obtém os melhores resultados multilabel-aware:
`in_set_accuracy` de 0.8944 e `top_2_accuracy` de 0.9107. A LR + TF-IDF de
produção permanece em 0.8476 e 0.8927, respectivamente; sua `in_set_accuracy`
fica 23,23 pontos percentuais acima da própria accuracy tradicional.

A run anterior do BERT (`5ee02745`) não tinha esses valores porque o script
experimental registrava apenas métricas agregadas single-label e não
preservava o checkpoint nem as saídas por amostra. Após a correção, a nova run
`4fd2bfc2` aplica as mesmas rotinas multirrótulo de
`src/evaluation/evaluate.py`, salva modelo, tokenizer, probabilidades e
métricas em `models/bert_experiment/`, e registra os resultados no MLflow.

## Por que essas 4 (e não as outras 3 testadas)

Além destes 4, também foram testados e descartados da apresentação final:

- **Complement Naive Bayes** (`alpha=2.0` tunado, CV 0.6067 ± 0.0048) e
  **Linear SVM calibrado** (`C=0.05` tunado, CV 0.5963 ± 0.0095) — ambos
  lineares em TF-IDF, redundantes com a história que a Regressão Logística já
  conta (linear bate não-linear neste corpus esparso); ver
  `docs/model_card.md` para a tabela completa dos 3 lineares comparados.
- **Random Forest** (test F1 0.487, sem tuning) — pior resultado entre todos
  os candidatos, sem ganho didático adicional ao que Gradient Boosting já
  ilustra sobre não-lineares em TF-IDF esparso.

Os 4 escolhidos cobrem os quatro eixos de decisão mais relevantes para
demonstrar evolução: modelo linear tunado por CV (baseline forte), modelo
não-linear (contraste), representação semântica alternativa (embeddings) e
teto de deep learning (BERT) — sem repetir a mesma conclusão duas vezes.

## Decisão de produção

O modelo em produção continua sendo **Logistic Regression + TF-IDF (C=0.3)**
(experimento 1), promovido pelo critério `cv_macro_f1_mean` + desempate por
`cv_macro_f1_std`. Embora o BERT genérico tenha vencido em F1/accuracy de
teste, ele não substitui a produção nesta entrega porque:

1. o resultado é um ponto único sem validação cruzada (menos robusto
   estatisticamente que os 3 primeiros, que foram comparados sob o mesmo
   critério de CV);
2. o artefato é substancialmente maior (cerca de 438 MB), requer runtime de
   transformer e, para reproduzir o throughput medido, infraestrutura com GPU;
   a medição batelada não demonstra vantagem de latência online sobre o pipeline
   sklearn + TF-IDF;
3. o objetivo desta fase era validar hipóteses de modelagem (plano
   `docs/plano_melhoria_f1.md`), não migrar a arquitetura de serving.

BERT/biomédico permanecem candidatos documentados para uma fase futura caso o
ganho de F1 justifique o custo de infraestrutura — próximos passos (reformular
o problema como multi-rótulo, avaliar BERT biomédico como `PubMedBERT` com CV
completo) seguem descritos em `docs/plano_melhoria_f1.md` (itens P1-P3).
