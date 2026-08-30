# Model Card — Medical Text Classifier

## Finalidade

Classificar abstracts públicos em inglês em cinco categorias amplas:

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
`medical_tc_labels.csv` do Medical Abstracts TC Corpus. Os textos são abstracts
públicos em inglês; não são prontuários do projeto e não devem conter
identificadores de pacientes.

O treino oficial é dividido de forma estratificada em 90% para fit e 10% para
validação, usando `random_state=42`. O teste oficial permanece integral e é
avaliado somente depois do treino. A origem, versão, quantidade final de linhas,
licença e eventuais restrições de redistribuição devem ser confirmadas na fonte
do corpus antes de publicar uma entrega.

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

- **padrão: Regressão Logística balanceada** (`C=0.3`, escolhida por busca de
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
(`config.tuning.grid`) com validação cruzada estratificada
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

Resultado da comparação (3-fold CV sobre treino+validação, grade reduzida:
`C`/`alpha` com `class_weight=balanced` fixo):

| Modelo | CV F1-macro (média ± desvio) | Test F1-macro | Test recall minorias |
| --- | --- | --- | --- |
| **Logistic Regression (C=0.3)** | **0.6165 ± 0.0023** | **0.6169** | **0.735** |
| Complement NB (alpha=2.0) | 0.6067 ± 0.0048 | 0.6070 | 0.660 |
| Linear SVM calibrado (C=0.05) | 0.5963 ± 0.0095 | 0.6064 | 0.512 |

## Avaliação

`make evaluate` grava `metrics/eval_metrics.json` com seções `validation` e
`test`. Cada seção contém accuracy, precision/recall/F1 por classe, métricas
macro e weighted, recall médio das duas classes minoritárias, labels e matriz de
confusão.

A seção `test` também traz três métricas "honestas" que contextualizam o
F1-macro diante de um corpus multi-rótulo achatado (ver
`docs/plano_melhoria_f1.md`):

- **`in_set_accuracy`** — fração de previsões que pertencem ao conjunto
  completo de rótulos válidos daquele abstract no corpus (um abstract pode
  aparecer sob mais de uma condição). Sempre ≥ accuracy exata.
- **`accuracy_by_label_count`** — accuracy exata, estratificada por quantos
  rótulos válidos o abstract carrega (`"1"`, `"2"`, `"3"`, `"4"`). Cai
  fortemente conforme o abstract tem mais rótulos possíveis, porque o
  gabarito escolheu só um deles.
- **`top_2_accuracy`** — fração em que o rótulo verdadeiro está entre as duas
  classes de maior probabilidade prevista; relevante porque o uso declarado é
  apoio à priorização, não decisão automática, e um top-2 já é acionável
  nesse contexto.

Essas três métricas não substituem o F1-macro reportado — apenas mostram, com
número, o que ele não consegue medir neste corpus especificamente.

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

- **O corpus é multi-rótulo achatado em multiclasse**: 26% dos abstracts
  aparecem mais de uma vez no dataset original, cada ocorrência com uma
  condição diferente igualmente válida — o mesmo texto pode legitimamente ser
  `neoplasms` numa linha e `general pathological conditions` em outra. Um
  oráculo que sorteia entre os rótulos válidos de cada abstract atinge
  F1-macro ≈ 0.78 no teste oficial; esse é o teto real de um classificador
  single-label neste corpus, não um limite do modelo ou das features.
  Diagnóstico completo, incluindo a decomposição do erro por número de
  rótulos e a proposta de reformulação como multi-rótulo, em
  `docs/plano_melhoria_f1.md`.
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
