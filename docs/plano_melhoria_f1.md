# Plano de melhoria de F1 — Medical Text Classifier

Documento de diagnóstico e propostas para elevar F1-macro, precision e recall
do classificador de abstracts médicos. Todos os números abaixo foram medidos
neste repositório, sobre os splits em `data/processed/`, e são reproduzíveis
pelos scripts descritos no Apêndice.

> **Status:** diagnóstico histórico — descreve o estado single-label que
> motivou as mudanças. A frente **P0** (tuning + CV, §1 item 1) e a frente
> **P3** (reformular como multirrótulo, §1 item 3) já foram implementadas em
> produção (`src/models/classifier.py`, `OneVsRestClassifier`; ver
> `docs/model_card.md` e `notebooks/01_eda_nlp_hospital.ipynb` §7 para a
> análise que motivou a decisão). A frente **P2** (transformer biomédico) foi
> testada como baseline experimental (`scripts/finetune_bert.py`), documentada
> em `docs/metodologia_experimentos.md`, e não substituiu a produção. Os
> números de F1/teto teórico abaixo são do modelo single-label anterior à
> migração — não descrevem o pipeline atual.

---

## 1. Resumo executivo

O F1-macro está travado em ~0.60 há seis experimentos. A investigação mostra
que isso **não é falta de modelo, de features ou de dados**: é uma propriedade
do próprio corpus.

**O Medical Abstracts TC Corpus é um dataset multirrótulo achatado em
multiclasse.** O mesmo abstract aparece em várias linhas, uma por condição
aplicável, cada uma com um rótulo diferente. 26% dos abstracts (2.929 de
11.227) carregam de 2 a 4 rótulos distintos. O treino ensina o modelo a
responder `A` para um texto e o teste cobra `B` para o mesmo texto.

Consequências medidas:

| Fato medido | Valor |
| --- | --- |
| Linhas totais / abstracts únicos | 14.438 / 11.227 |
| Abstracts com mais de 1 rótulo | 2.929 (26,1%) |
| Pares `(texto, rótulo)` repetidos | 0 |
| Linhas de teste cujo abstract já está no treino | 1.010 (35,0%) |
| ...destas, com o rótulo correto ausente do treino | 1.010 (100%) |
| Acurácia do modelo atual em abstracts de 1 rótulo | 0,736 |
| Acurácia do modelo atual em abstracts de 2 rótulos | 0,414 |
| Acurácia do modelo atual em abstracts de 3 rótulos | 0,272 |
| Predição pertence ao conjunto verdadeiro de rótulos | 0,814 |
| **Teto de F1-macro de um oráculo perfeito** | **0,780** |

Ou seja: **22% do conjunto de teste é classificado corretamente e contado como
erro**, porque o gabarito escolheu outro rótulo igualmente válido do mesmo
abstract. O modelo atual entrega 0,61 num problema cujo máximo teórico é 0,78.

Isso muda o objetivo do trabalho. Há três frentes, nesta ordem de retorno:

1. **Corrigir o método de avaliação e seleção** — ganho medido de **+0,011
   F1-macro no teste** (0,608 → 0,619) só com tuning e validação cruzada,
   sem trocar de modelo. Custo: horas.
2. **Trocar a família de modelo** por um transformer biomédico — é a única
   alavanca capaz de atacar os 26 pontos de acurácia que faltam no subconjunto
   de rótulo único (0,736 hoje). Custo: dias.
3. **Reformular a tarefa como multirrótulo** — não sobe o F1 do benchmark, mas
   produz um sistema honesto, mais útil clinicamente e defensável na
   apresentação. Custo: dias.

---

## 2. Ponto de partida: o que o MLflow registra hoje

Runs do experimento `Medical Text Classifier API` (`mlflow.db`):

| Run | val F1-macro | test F1-macro | test recall minorias |
| --- | --- | --- | --- |
| `complement_nb-tfidf` (**em produção**) | 0,6063 | 0,6101 | 0,663 |
| `logistic_regression-embeddings` | 0,6111 | 0,6067 | 0,728 |
| `logistic_regression-tfidf` | 0,6104 | 0,6025 | 0,686 |
| `gradient_boosting-tfidf` | 0,5627 | 0,5188 | 0,297 |
| `linear_svm-tfidf` | 0,5231 | 0,5369 | 0,384 |
| `random_forest-tfidf` | 0,5006 | 0,4871 | 0,428 |

Três leituras importantes desta tabela:

- Os três primeiros estão dentro do **ruído um do outro**. A validação tem
  1.155 linhas; o desvio-padrão observado entre folds naquela CV 5-fold
  exploratória era de ±0,007 a
  ±0,011. Diferenças de 0,005 não significam nada.
- Todos os runs usaram **hiperparâmetros default**: `C=1.0`, `alpha=1.0`,
  `n_estimators=160`. Nenhuma busca foi feita.
- Por causa disso, o ranking está errado (ver §3.6).

Matriz de confusão do modelo em produção (validação), com o problema visível na
última linha:

```
                                 neo  dig  ner  car  gpc
neoplasms                        213   11    9    7   13
digestive system diseases         17   79    4    5   15
nervous system diseases           20    3   89   14   28
cardiovascular diseases            6    6    9  205   18
general pathological conditions   76   47   58   81  122   <-- recall 0,318
```

`general pathological conditions` tem 33% do dataset, recall de 0,318 e F1 de
0,42. Sozinha, ela derruba o F1-macro em cerca de 4 pontos.

---

## 3. Diagnóstico

### 3.1 O corpus é multirrótulo achatado

```
n_rótulos distintos por abstract único:
  1 rótulo  -> 8.298 abstracts
  2 rótulos -> 2.653
  3 rótulos ->   270
  4 rótulos ->     6
pares (texto, rótulo) duplicados: 0
```

Nenhum par `(texto, rótulo)` se repete, e todo texto repetido tem rótulos
diferentes. Isso é a assinatura exata de uma tabela multirrótulo normalizada em
formato longo — não é sujeira de coleta, é o formato do corpus.

Exemplo real do treino, o abstract sobre síndrome do encarceramento após
doença viral, que aparece três vezes com os rótulos
`general pathological conditions`, `neoplasms` e `nervous system diseases`.
Os três estão corretos. O modelo só pode acertar um.

### 3.2 O teto matemático é 0,78

Simulando um oráculo que conhece o conjunto verdadeiro de rótulos de cada
abstract e sorteia um deles (200 repetições):

```
F1-macro = 0,7796 ± 0,0059   (acurácia 0,7832)
```

**Nenhum modelo single-label pode passar disso de forma legítima.** O gap de
0,61 → 0,78 é a fronteira real de trabalho; os 0,22 restantes são inalcançáveis
por construção.

### 3.3 Decomposição do erro

Modelo em produção, no teste oficial:

| Subconjunto | n | Acurácia |
| --- | --- | --- |
| Abstracts com 1 rótulo | 1.691 | **0,736** |
| Abstracts com 2 rótulos | 1.032 | 0,414 |
| Abstracts com 3 rótulos | 162 | 0,272 |
| Abstracts com 4 rótulos | 3 | 0,000 |
| **Total** | 2.888 | 0,594 |

`predição ∈ conjunto verdadeiro de rótulos` = **0,814**.

O modelo já entende a medicina muito melhor do que o F1-macro de 0,60 sugere.
Quando erra, na maioria das vezes escolheu uma condição que o abstract
realmente discute.

**Onde há espaço real:** o subconjunto de rótulo único, hoje em 0,736. Se ele
subir, o número global sobe:

| Acurácia em k=1 | Acurácia global projetada |
| --- | --- |
| 0,736 (hoje) | 0,594 |
| 0,80 | 0,632 |
| 0,85 | 0,661 |
| 0,90 | 0,690 |

Meta realista para este projeto: **F1-macro de 0,65 a 0,68**.

### 3.4 `general pathological conditions` é um rótulo transversal, não uma classe

| Rótulo | Abstracts | % que também têm outro rótulo |
| --- | --- | --- |
| general pathological conditions | 4.805 | **50,2%** |
| digestive system diseases | 1.494 | 53,2% |
| nervous system diseases | 1.925 | 45,5% |
| cardiovascular diseases | 3.051 | 35,7% |
| neoplasms | 3.163 | 30,6% |

Pares mais frequentes: `cardiovascular + gpc` (870), `gpc + neoplasms` (619),
`digestive + gpc` (603), `gpc + nervous` (591).

Metade das ocorrências de `general pathological conditions` convive com uma
condição de sistema específico. A classe não é disjunta das outras — é uma
dimensão ortogonal. Forçá-la ao softmax é o que produz o recall de 0,318.

### 3.5 O platô não é de features nem de volume de dados

**Varredura do espaço de features** (48 combinações de `max_features`,
`min_df`, `sublinear_tf`, `ngram_range`, com LogReg fixo):

```
melhor: 0,6054   (max_features=5000, min_df=5, sublinear=False, 1-gram)
pior:   0,5780   (sem cap, min_df=2, sublinear=True, 1-2 gram)
```

Amplitude total: 0,027. O `max_features=5000` já configurado está no topo.
Aumentar vocabulário **piora** (193k features → 0,578). Não há ganho aqui.

**Curva de aprendizado** (CNB, F1-macro no teste):

```
 25% do treino (n=2.599):  0,5738
 50% do treino (n=5.198):  0,5802
 75% do treino (n=7.796):  0,5904
100% do treino (n=10.395): 0,5870
```

Chapada a partir de 50%. **Mais dados do mesmo tipo não resolvem.**

**Pré-processamento spaCy: mantenha.** É um dos poucos itens que
comprovadamente ajuda:

| Entrada | CNB (test F1) | LogReg (test F1) |
| --- | --- | --- |
| Texto cru | 0,5870 | 0,5873 |
| Lemmatização + stopwords (atual) | **0,6092** | 0,6025 |

### 3.6 A comparação de modelos no MLflow está viciada

Resultado exploratório histórico com validação cruzada 5-fold sobre treino+val
(n=11.550), texto pré-processado. Este quadro não é o protocolo oficial atual,
que usa 3 folds e está registrado em `docs/metodologia_experimentos.md`:

| Configuração | CV F1-macro | Test F1-macro |
| --- | --- | --- |
| **LinearSVC(C=0.1, balanced)** | 0,6194 ± 0,0100 | **0,6195** |
| **LogReg(C=0.3, balanced)** | **0,6196 ± 0,0063** | 0,6161 |
| CNB(alpha=2.0) | 0,6065 ± 0,0113 | 0,6066 |
| CNB(alpha=1.0) — *produção* | 0,6033 ± 0,0105 | 0,6081 |
| LogReg(C=1.0) — *config atual* | 0,6100 ± 0,0086 | 0,6055 |
| LinearSVC(C=1.0) — *config atual* | 0,5345 ± 0,0078 | — |
| LogReg(C=3.0) | 0,5781 ± 0,0082 | — |
| LogReg(C=10.0) | 0,5244 ± 0,0069 | — |

Duas conclusões:

1. **O Linear SVM não é ruim — o `C` estava errado.** Com `C=1.0` marca 0,534;
   com `C=0.1` marca 0,619. O run `linear_svm-tfidf` (0,5369) descartou o
   melhor modelo do repositório por causa de um hiperparâmetro default.
2. **O ComplementNB venceu por sorte amostral.** Ele é o pior dos três lineares
   sob CV, mas ganhou na validação de 1.155 linhas. O `registry.metric` é
   `validation_macro_f1` — o critério de promoção está premiando ruído.

**Alerta sobre ajuste de limiar.** Testei otimizar bias por classe para
maximizar F1-macro diretamente na validação. Ganha na validação e perde no
teste:

| Modelo | val (antes → depois) | test (antes → depois) |
| --- | --- | --- |
| CNB | 0,5834 → 0,6117 | 0,5870 → 0,6001 |
| LogReg | 0,6019 → 0,6074 | 0,5873 → **0,5782** |
| Ensemble | 0,5924 → 0,6037 | 0,5908 → **0,5805** |

Com 1.155 linhas, o ajuste memoriza a validação. Se for feito, tem de ser
**dentro de CV** (§P2.2), nunca no split único.

---

## 4. Propostas

### P0 — Correções de método (ganho já medido, custo de horas)

#### P0.1 · Buscar hiperparâmetros de verdade

**Ganho medido: +0,011 F1-macro no teste (0,6081 → 0,6195).**

Nenhum dos seis runs tunou nada. Adicionar uma etapa de busca (`GridSearchCV`
ou `RandomizedSearchCV` com `scoring="f1_macro"`, `StratifiedKFold(5)`) sobre:

```yaml
tuning:
  enabled: true
  cv_folds: 3
  scoring: f1_macro
  grid:
    linear_svm:
      C: [0.03, 0.05, 0.1, 0.2, 0.3]
    logistic_regression:
      C: [0.1, 0.2, 0.3, 0.5, 1.0]
    complement_nb:
      alpha: [0.5, 1.0, 2.0, 4.0]
    tfidf:
      max_features: [3000, 5000, 10000]
      ngram_range: [[1, 1], [1, 2]]
```

Arquivos: novo `src/training/tuning.py`, chamado por `trainer.py`; grid em
`config/config.yaml`; melhores params logados como params do run MLflow.

#### P0.2 · Trocar o critério de promoção para CV

**Ganho: elimina promoções por ruído.**

`config/config.yaml` → `registry.metric: validation_macro_f1` deve virar
`cv_macro_f1_mean`, com desempate por `cv_macro_f1_std` ascendente. O trainer
passa a logar `cv_macro_f1_mean` e `cv_macro_f1_std`. `src/models/promote.py`
seleciona por essa métrica.

Sem isso, qualquer melhoria futura corre o risco de ser descartada por uma
validação de 1.155 linhas.

#### P0.3 · Ajustar o `class_weight` para o alvo certo

**Ganho esperado: +0,005 a +0,015, a confirmar.**

`class_weight="balanced"` otimiza recall equilibrado, não F1-macro. Vale
comparar, dentro da mesma CV, `balanced` × `None` × pesos customizados que
penalizem menos a classe `general pathological conditions` — hoje ela tem 33%
do volume e é a que mais sangra.

#### P0.4 · Treinar o artefato final em treino + validação

**Ganho esperado: marginal, mas gratuito.**

Depois de escolher o modelo por CV, refit em `train + validation` (11.550 em
vez de 10.395) antes de persistir. O teste oficial permanece intocado, como
exige o `AGENTS.md`. Já é assim nos números da tabela do §3.6.

---

### P1 — Salto de capacidade (custo de dias, maior retorno)

#### P1.1 · Transformer biomédico — a proposta principal

**Ganho esperado: +0,04 a +0,08 F1-macro. Meta: 0,65–0,68.**

É a única alavanca que ataca o subconjunto de rótulo único (0,736 → 0,85+).
TF-IDF vê palavras isoladas; abstracts médicos dependem de contexto, negação e
relações entre entidades ("metastatic lesion secondary to colonic
adenocarcinoma" é neoplasia, não doença digestiva).

Modelos candidatos, todos pré-treinados em PubMed — exatamente este domínio:

| Modelo | Observação |
| --- | --- |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | Primeira escolha: vocabulário treinado do zero em abstracts PubMed |
| `michiyasunaga/BioLinkBERT-base` | Alternativa forte, usa links entre documentos |
| `dmis-lab/biobert-base-cased-v1.2` | Mais antigo, referência de comparação |

**Viabilidade local confirmada:** há uma RTX 4070 Laptop (8 GB) nesta máquina.
Com `max_length=512`, `batch_size=16`, gradient checkpointing e 3–4 épocas em
10.395 exemplos, o fine-tuning roda em 15–25 minutos. Não precisa de nuvem.

Cuidados de integração com a arquitetura existente:

- **Dependência pesada.** `torch` + `transformers` inflam a imagem em ~2,5 GB.
  Use o mesmo padrão já aplicado ao MLflow no `pyproject.toml`: um grupo
  `[tool.poetry.group.train.dependencies]`. O stage `api` do Dockerfile
  continua sem `torch`.
- **Serving.** O contrato `TriagePredictor` (`backend` + `predict(text) ->
  (label, scores)`) já permite um terceiro backend. Exporte via `optimum` para
  ONNX com quantização int8 e sirva por `onnxruntime`, que já é dependência.
  Assim o stage `api` não muda de stack.
- **Latência.** O benchmark atual é p95 = 4,2 ms (ONNX) contra 6,3 ms
  (sklearn). Um BERT-base int8 em CPU fica na casa de 40–120 ms para 512
  tokens. Isso **precisa** ser medido com `make benchmark-latency` e discutido
  no `docs/architecture.md`: é um trade-off explícito de +0,05 F1 por ~30x de
  latência. Se o requisito de latência for rígido, mantenha o modelo linear
  como default e o transformer como backend opcional via `MODEL_BACKEND` — o
  que, aliás, dá uma comparação de otimização muito mais rica para a rubrica
  do desafio.

Arquivos: novo `src/models/transformer_classifier.py` (nova Strategy no
`ModelFactory`), `src/optimization/export_onnx.py` (branch para transformer),
`src/serving/model_loader.py` (terceiro backend), `config/config.yaml`
(`model.type: biomedbert` + bloco de hiperparâmetros).

#### P1.2 · Reformular como multirrótulo (binary relevance)

**Ganho no benchmark: neutro (0,606 val / 0,600 test, dentro do ruído).
Ganho real: um sistema correto.**

Medi a versão ingênua — deduplicar o treino por texto, unir os rótulos, treinar
5 classificadores binários e fazer argmax. O F1-macro fica igual ao atual,
porque o gabarito continua achatado. **Mas:**

- Elimina o sinal contraditório do treino (hoje o mesmo texto aparece com
  rótulos diferentes, o que é ruído puro para a loss).
- Permite responder o que o problema realmente pede: *quais* condições este
  abstract aborda, com uma probabilidade por condição.
- Resolve `general pathological conditions` de forma natural: ela vira um
  rótulo que pode coexistir, em vez de competir no softmax.
- A API já retorna `scores` por classe. Basta adicionar um campo `labels` com
  os rótulos acima do limiar, mantendo `label` (argmax) para compatibilidade.

**Recomendação:** implementar como modo alternativo (`task: multilabel` no
YAML), reportar as duas métricas, e usar isso como argumento central na
apresentação — mostra domínio do problema, não só do ferramental.

---

### P2 — Refinamentos (fazer depois de P0 e P1)

#### P2.1 · Métricas honestas ao lado do F1-macro

Adicionar em `src/evaluation/evaluate.py` e no `eval_metrics.json`:

- **`in_set_accuracy`** — fração de predições que pertencem ao conjunto
  verdadeiro de rótulos do abstract. Modelo atual: **0,814** contra acurácia
  exata de 0,594.
- **`accuracy_by_label_count`** — acurácia estratificada por k (§3.3).
- **`top_2_accuracy`** — relevante porque o uso declarado é priorização, e um
  top-2 é acionável nesse contexto.

Isso não maquia o resultado: mostra, com número, o que o F1-macro não consegue
medir neste corpus. Deve ir para `docs/model_card.md`.

#### P2.2 · Otimização de limiar **dentro** de CV

Só depois de P0.2. Ajustar bias por classe usando predições out-of-fold
(`cross_val_predict`) em 11.550 linhas, não no split de 1.155. Ganho esperado
de +0,01 a +0,02 no recall das minorias — mas apenas se o ajuste for feito
sobre folds, já que a versão ingênua **perdeu 0,009 no teste** (§3.6).

#### P2.3 · Ensemble

Testei média de scores padronizados de CNB + LogReg + LinearSVC: 0,5908 no
teste, **pior** que o melhor componente isolado. Ensemble de modelos que erram
junto não ajuda. Só vale reconsiderar depois de P1.1, combinando transformer +
linear, que têm vieses genuinamente diferentes.

---

### P3 — O que **não** fazer

#### P3.1 · Não explorar o vazamento estrutural

Como 100% dos 1.010 abstracts de teste presentes no treino têm o rótulo correto
**ausente** do treino, dá para excluir os rótulos já vistos e subir o F1 de
0,587 para **0,711** — 12 pontos de graça.

**Isso é exploração de artefato de benchmark, não modelagem.** Não generaliza
para produção, onde não existe "rótulo já visto para este texto". Documente o
número em `docs/model_card.md` como evidência do problema do corpus, e não o
use no artefato entregue.

#### P3.2 · Não insistir em modelos de árvore

Random Forest (0,487) e Gradient Boosting (0,519) estão 10 pontos atrás dos
lineares em TF-IDF esparso de alta dimensão. É o comportamento esperado. Não
vale gastar tempo tunando.

#### P3.3 · Não aumentar o vocabulário

Medido: 193k features derrubam o F1 para 0,578. `max_features=5000` já é ótimo.

---

## 5. Roadmap sugerido

| Etapa | Entrega | Ganho esperado | Esforço |
| --- | --- | --- | --- |
| 1 | P0.2 — critério de promoção por CV | base confiável | 2–3 h |
| 2 | P0.1 + P0.3 + P0.4 — tuning e refit | **+0,011 medido** | 4–6 h |
| 3 | P2.1 — métricas honestas + model card | clareza | 3–4 h |
| 4 | P1.1 — transformer biomédico | **+0,04 a +0,08** | 2–3 dias |
| 5 | P1.1b — ONNX int8 + benchmark de latência | trade-off documentado | 1 dia |
| 6 | P1.2 — modo multirrótulo | correção conceitual | 1–2 dias |
| 7 | P2.2 — limiar dentro de CV | +0,01 a +0,02 | 4 h |

Alvo ao fim da etapa 5: **F1-macro de 0,65 a 0,68 no teste oficial**, com
precision e recall macro acompanhando (hoje 0,603 / 0,669).

Se o tempo for curto, as etapas 1–3 sozinhas já entregam um resultado melhor,
um método defensável e a análise que diferencia o trabalho.

---

## 6. Protocolo de medição

Para que as próximas comparações valham alguma coisa:

1. **Toda decisão de modelo sai de CV 3-fold estratificada** sobre treino+val
   (n=11.550), nunca do split de validação isolado.
2. **Reportar média ± desvio entre folds.** Diferenças menores que 2× o desvio
   (≈0,015) não são conclusão — são ruído.
3. **O teste oficial é tocado uma vez por candidato**, depois da decisão. Não é
   critério de seleção.
4. **Logar no MLflow**: `cv_macro_f1_mean`, `cv_macro_f1_std`, `n_folds`, os
   hiperparâmetros vencedores da busca, `in_set_accuracy` e
   `accuracy_by_label_count`.
5. **Nomear os runs com a variante completa**, incluindo hiperparâmetros
   tunados — hoje seis runs se chamam `logistic_regression-tfidf` e três estão
   sem métrica nenhuma (dois em `RUNNING`, um `FINISHED` vazio). Vale limpar.

---

## 7. Riscos e impactos arquiteturais

| Risco | Mitigação |
| --- | --- |
| `torch` infla a imagem Docker | Grupo Poetry `train`, como já é feito com MLflow; stage `api` serve só ONNX |
| Latência do transformer quebra o requisito real-time | Medir com `make benchmark-latency`; manter o linear como default e o transformer como `MODEL_BACKEND` opcional |
| Export ONNX hoje converte só o classificador final | Branch específico para transformer via `optimum`; atualizar `export_onnx.py`, `model_loader.py`, testes e docs juntos |
| Teste de arquitetura exige funções ≤ 20 linhas lógicas | Quebrar tuning e treino do transformer em funções pequenas, no padrão Strategy/Factory já usado |
| Mudança de contrato da API no modo multirrótulo | Adicionar `labels` sem remover `label`; `POST /predict` continua compatível |
| Ganho some ao mudar de split | Nunca voltar a decidir por validação única (§P0.2) |

---

## 8. Apêndice — como reproduzir os números

Os diagnósticos deste documento vieram de scripts descartáveis. Vale
promovê-los a `scripts/analyze_corpus.py` para ficarem versionados:

```python
# estrutura multirrótulo
all_ = pd.concat([train, validation, test], ignore_index=True)
sets = all_.groupby("text").label.agg(set)
sets.apply(len).value_counts()                  # 8298 / 2653 / 270 / 6

# teto teórico
rng = np.random.default_rng(0)
picks = [rng.choice(sorted(sets[t])) for t in test.text]
f1_score(test.label, picks, average="macro")    # 0,7796

# decomposição do erro
k = test.text.map(sets.apply(len))
accuracy_by(k)                                  # 0,736 / 0,414 / 0,272

# in-set accuracy
np.mean([p in sets[t] for p, t in zip(pred, test.text)])   # 0,814
```

Ambiente: use `.venv/Scripts/python.exe` — é o interpretador que tem spaCy e os
modelos `en_core_web_sm` / `en_core_sci_md` instalados.

---

## 9. Uma nota sobre o que isso significa para o Tech Challenge

O `AGENTS.md` deixa claro que o sistema é apoio à priorização, não diagnóstico.
A descoberta do §3.1 reforça isso de forma concreta: um abstract médico
raramente pertence a uma única categoria, e um sistema que finge o contrário
está estruturalmente errado, mesmo com F1 alto.

Levar essa análise para a documentação e para o vídeo STAR vale mais do que
alguns pontos de métrica. Mostra diagnóstico de dados, teto teórico, decisão
consciente de não explorar vazamento e trade-off explícito entre acurácia e
latência — que é exatamente o repertório que a rubrica de "Modelagem e
otimização" procura.
