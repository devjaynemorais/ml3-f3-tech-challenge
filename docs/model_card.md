# Model Card — Medical Text Classifier

## Finalidade

Classificar abstracts públicos em inglês em uma ou mais de cinco categorias:

1. `neoplasms`
2. `digestive system diseases`
3. `nervous system diseases`
4. `cardiovascular diseases`
5. `general pathological conditions`

O artefato é uma demonstração técnica de NLP/MLOps e pode apoiar organização ou
priorização documental sob supervisão. Não é um diagnóstico, não mede urgência
clínica e não substitui profissionais de saúde.

## Dados

O pipeline espera os arquivos `medical_tc_train.csv`, `medical_tc_test.csv` e
`medical_tc_labels.csv` do Medical Abstracts Text Classification Corpus,
publicado por Schopf, Braun e Matthes no
[repositório dos autores](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
e espelhado no
[Kaggle](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus).
São 14.438 linhas rotuladas em inglês (11.550 no treino oficial e 2.888 no
teste), correspondentes a 11.227 abstracts únicos. Os textos são abstracts
públicos; não são prontuários do projeto e não devem conter identificadores de
pacientes.

O corpus processado é distribuído sob **CC BY-SA 3.0**. O uso deve atribuir os
autores e preservar a mesma licença em redistribuições ou derivados. Referência:
Schopf, Braun e Matthes, *Evaluating Unsupervised Text Classification:
Zero-Shot and Similarity-Based Approaches*, DOI
[`10.1145/3582768.3582795`](https://doi.org/10.1145/3582768.3582795).

Treino e teste são agregados separadamente por abstract. Se um texto aparece
nos dois CSVs oficiais, ele é removido somente do treino; nenhum label atravessa
a fronteira. Isso elimina 988 grupos/1.097 linhas brutas do treino e deixa
8.457 abstracts para treino+validação e 2.770 no teste, com sobreposição zero.

## Preprocessing e features

O transformer persistido aplica, na ordem configurada:

1. normalização Unicode, lowercase e espaços;
2. remoção de pontuação com preservação de letras e números;
3. tokenização e lematização com `en_core_web_sm 3.8.0`;
4. remoção de stopwords NLTK e tokens de um caractere.

Em seguida, `features.type` seleciona a representação:

- `tfidf` (padrão): até 5.000 features, n-gramas de 1 a 2, `min_df=2`;
- `embeddings`: vetor médio de palavras do `en_core_sci_md` (scispaCy,
  treinado em texto biomédico) — captura similaridade semântica
  (ex.: "carcinoma" ~ "tumor") que o TF-IDF não vê, à custa de ~40-50% mais
  latência por requisição.

O mesmo pipeline de preprocessing é reutilizado na inferência, evitando
training-serving skew.

## Modelos

- **padrão: One-vs-Rest com Regressão Logística balanceada**, uma decisão
  binária independente por categoria, escolhida por busca de
  hiperparâmetros com validação cruzada — ver seção seguinte; melhor
  F1-macro, mais estável entre folds e melhor recall de classes minoritárias
  entre os candidatos comparados);
- alternativa: Complement Naive Bayes (`alpha=2.0` tunado; competitivo, mas
  abaixo da Regressão Logística após o tuning de ambos);
- alternativa: Linear SVM calibrado (Platt scaling via `CalibratedClassifierCV`
  — mesmo com `C` tunado (`0.05`), ficou abaixo dos dois acima; a calibração
  por CV parece distorcer o efeito do `class_weight=balanced`, custo que o
  LinearSVC não-calibrado do artigo de referência não paga, mas que é
  obrigatório aqui porque a API exige `predict_proba`);
- alternativa: Random Forest e Gradient Boosting (bem abaixo dos lineares em
  TF-IDF esparso de alta dimensão — não valem tuning, ver
  `docs/plano_melhoria_f1.md`);
- feature alternativa: embeddings biomédicos (`en_core_sci_md`), testados só
  com Regressão Logística — melhora recall de minorias mas não venceu no
  critério final (ver `docs/plano_melhoria_f1.md`).

Uma única Strategy é treinada por execução, escolhida em `config/config.yaml`.
A comparação entre Strategies é feita via `make train` + `make evaluate` para
cada `model.type`, com os resultados registrados no MLflow (`make mlflow`)
para comparar lado a lado.

### Busca de hiperparâmetros e critério de promoção (CV)

Com `tuning.enabled: true` (padrão), `make train` roda uma busca em grade
(`config.tuning.grid`) com validação cruzada por abstracts já agrupados
(`config.tuning.cv_folds` folds) sobre **treino + validação combinados**,
otimizando `f1_macro`. O pipeline vencedor já sai re-treinado (refit) em todo
esse conjunto — por isso `metadata["refit_includes_validation"] = true` e a
seção `validation` de `eval_metrics.json` deixa de ser held-out (vira um
resumo das estatísticas de CV, não uma reavaliação enviesada).

O critério de promoção no MLflow Model Registry (`registry.metric`) é
`cv_macro_f1_mean`, com desempate por `cv_macro_f1_std` (menor variância entre
folds). Isso substitui o critério anterior de comparar por uma métrica de
split único: com ~1.150 linhas de validação, diferenças abaixo de ~0.015 são
ruído estatístico, não sinal — ver `docs/plano_melhoria_f1.md` seção 3.6 para
o caso real em que isso invalidou uma comparação anterior.

Os números vigentes são gerados por `make experiments` em
`metrics/experiment_comparison.json`; a demo lê esse arquivo diretamente para
evitar tabelas desatualizadas.

#### Piso de qualidade e desempate por latência (`registry.min_metric`/`latency_metric`)

`make promote` (`src/utils/mlflow_tracking.py::find_best_model_run`) aplica
três critérios em sequência, nesta ordem — cada um existe para responder uma
pergunta de negócio distinta, não é só ajuste estatístico:

1. **Piso de qualidade — `registry.min_metric: 0.75`.** Nenhum run abaixo
   disso é sequer candidato a promoção, mesmo que seja o único disponível ou
   o mais rápido — a promoção falha alto (erro) em vez de subir um modelo
   ruim silenciosamente. **Por quê 0.75:** é o piso que já exclui o Gradient
   Boosting (`cv_macro_f1_mean` 0.629 — documentado acima como não
   competitivo neste corpus) do pipeline automático de promoção, e fica
   perto o suficiente da produção atual (0.7768) para que só variações
   marginais de re-treino sejam aceitas sem revisão humana. Do ponto de
   vista de negócio: um classificador de triagem errado com frequência gera
   retrabalho de revisão manual e corrói a confiança de quem usa a
   ferramenta — isso custa mais caro do que qualquer economia de infra, e
   nenhuma vantagem de latência compensa. O piso também é uma rede de
   segurança operacional: se `config.yaml` for editado incorretamente (já
   aconteceu durante o desenvolvimento — ver histórico do repositório) e o
   retreino cair para um modelo muito pior, a promoção automática bloqueia
   em vez de substituir a produção por engano.
2. **Faixa de empate — `registry.accuracy_tolerance: 0.015`.** Entre os runs
   que passaram no piso, qualquer um com `cv_macro_f1_mean` até 0.015 abaixo
   do melhor é tratado como estatisticamente empatado (é o mesmo limiar de
   ruído do parágrafo anterior — não é um segundo número inventado). **Por
   quê importa para o negócio:** sem essa faixa, a escolha do modelo de
   produção poderia mudar entre re-treinos por puro ruído do split, tratando
   custo de infraestrutura como irrelevante mesmo quando dois modelos são,
   na prática, o mesmo modelo com sorte diferente na validação.
3. **Desempate — `registry.latency_metric: latency_onnx_mean_ms`.** Só entre
   os runs empatados na faixa acima, vence o de menor latência média no
   backend ONNX (o backend de produção — ver `MODEL_BACKEND=onnx`). **Por
   quê ONNX e não sklearn:** é o número que reflete o custo real de servir
   em produção. Latência menor significa mais requisições por segundo com o
   mesmo hardware (custo de infraestrutura) e resposta mais rápida para
   quem usa a ferramenta — um ganho "de graça" quando a qualidade já é
   estatisticamente a mesma. Runs sem essa métrica (ninguém rodou `make
   benchmark-latency` ainda) caem de volta no critério antigo, então isso
   nunca bloqueia uma promoção por falta de dado.

**O que este critério deliberadamente não faz:** não existe um teto máximo de
latência. Uma diferença real de acurácia (fora da faixa de ruído) sempre
vence, custe o que custar em milissegundos — poucos ms a mais por requisição
não mudam a experiência nem o custo de forma perceptível neste volume de
uso, mas um modelo pior classificando errado tem custo operacional real.
Travar por latência arriscaria rejeitar um modelo genuinamente melhor por um
motivo de infraestrutura que nem é o gargalo do sistema.

## Avaliação

`make evaluate` grava `metrics/eval_metrics.json` com subset accuracy, Hamming
loss, Jaccard por amostra, cardinalidade de labels, precision/recall/F1 por
classe e médias macro, micro e weighted.

**Métricas justas (priorizar estas duas):** `macro_avg.f1` (F1-macro
multirrótulo) é a métrica principal — trata cada classe com peso igual,
resiste ao desbalanceamento entre as 5 categorias. `jaccard_samples`
(accuracy multirrótulo por amostra, Godbole & Sarawagi 2004) é a métrica de
"accuracy" a reportar — mede a sobreposição média entre o conjunto previsto e
o verdadeiro por abstract. `subset_accuracy` (coincidência exata do conjunto
inteiro de rótulos) e `hamming_loss` (erros por decisão binária) contextualizam
mas não devem substituir as duas primeiras: `subset_accuracy` é a mais rígida
das quatro e tende a subestimar o desempenho real quando um abstract tem mais
de um rótulo válido. `scripts/compare_experiments.py` imprime `test_f1_macro`
e `ml_accuracy` (= `jaccard_samples`) lado a lado por experimento.

As métricas devem ser reportadas a partir do arquivo gerado na mesma versão do
artefato servido. Não há números fixos neste documento para evitar publicar
resultados desatualizados ou obtidos de outro split.

## Artefatos e ONNX

O pipeline sklearn, metadata e artefatos ONNX registram schema, modelo,
parâmetros, Strategies, classes e volume de treino. O loader exige exatamente as
cinco classes canônicas. Artefatos antigos de três classes são rejeitados e
mantêm a API em modo degradado.

No backend ONNX, preprocessing, TF-IDF e etapas de features permanecem no
prefixo joblib; somente o classificador é convertido. A paridade numérica é
testada com `rtol=1e-5` e `atol=1e-6`.

## Limitações e riscos

- **O corpus original é multi-rótulo achatado em linhas**: 26% dos abstracts
  aparecem mais de uma vez no dataset original, cada ocorrência com uma
  condição diferente igualmente válida — o mesmo texto pode legitimamente ser
  `neoplasms` numa linha e `general pathological conditions` em outra.
  O pipeline atual agrega esses labels dentro de cada split e treina um modelo
  multilabel. Labels que existem apenas nas linhas removidas do treino não são
  copiados para o teste; por isso o ground truth pode continuar incompleto.
- `general pathological conditions` concentra a maior parte dos erros do
  modelo, espalhados quase igualmente entre as outras 4 classes (sem par de
  confusão dominante) — consequência direta do ponto acima: é a classe mais
  associada a rótulos concorrentes (metade das suas ocorrências no corpus
  também carrega outra condição). Auditoria manual de amostras mal
  classificadas confirma: abstracts com conteúdo claramente cardiovascular,
  digestivo ou neurológico foram rotulados como "geral" no corpus original.
  Isso impõe um teto de recall nessa classe que técnicas de modelagem não
  resolvem sozinhas.
- As categorias são amplas e não representam diagnósticos específicos.
- Abstracts publicados diferem de notas clínicas, laudos e mensagens reais.
- Vocabulário, estilo, prevalência e idioma podem mudar entre fontes.
- Classes minoritárias exigem atenção a recall, macro-F1 e matriz de confusão;
  accuracy isolada pode esconder falhas relevantes.
- Probabilidades do classificador não devem ser interpretadas como risco
  clínico sem calibração e validação externa.
- Qualquer uso em ambiente de saúde exige revisão humana, controle de falsos
  negativos, rastreabilidade, privacidade, segurança e avaliação ética.

## Uso não recomendado

Não usar para diagnóstico autônomo, decisão terapêutica, priorização clínica
sem supervisão, classificação de textos fora do domínio ou processamento de
dados identificáveis sem governança apropriada.
