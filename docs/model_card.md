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

Em seguida, TF-IDF usa até 5.000 features, n-gramas de 1 a 2 e `min_df=2`. O
mesmo pipeline de preprocessing é reutilizado na inferência, evitando training-
serving skew.

## Modelos

- padrão: Regressão Logística balanceada;
- alternativa: Random Forest balanceada;
- alternativa: Gradient Boosting com seleção chi-quadrado e matriz densa.

Uma única Strategy é treinada por execução, escolhida em `config/config.yaml`.
O sistema não escolhe automaticamente o melhor modelo.

## Avaliação

`make evaluate` grava `metrics/eval_metrics.json` com seções `validation` e
`test`. Cada seção contém accuracy, precision/recall/F1 por classe, métricas
macro e weighted, recall médio das duas classes minoritárias, labels e matriz de
confusão.

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
