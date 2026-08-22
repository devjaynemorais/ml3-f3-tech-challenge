# SPEC - Notebook de EDA e Baseline para Classificacao de Textos Hospitalares

## 1. Objetivo

Construir um Jupyter Notebook autocontido para realizar uma analise exploratoria inicial dos dados, identificar necessidades de preprocessamento textual e elaborar uma arquitetura teorica inicial para um MVP de classificacao de textos usando NLP.

O notebook deve servir como base exploratoria para orientar a futura etapa de deploy, sem se tornar ainda o pipeline definitivo de producao.

## 2. Contexto do Projeto

O projeto envolve classificacao de textos para identificar possiveis doencas em um contexto hospitalar.

Nesta etapa inicial, o foco e:

- entender a estrutura e qualidade dos dados;
- identificar problemas de texto que exigem preprocessamento;
- avaliar uma arquitetura baseline simples;
- gerar insights iniciais sobre classes, distribuicoes e padroes textuais;
- documentar uma proposta teorica de arquitetura para evolucao do MVP.

## 3. Escopo

Todo o codigo necessario para as analises deve permanecer dentro do escopo do notebook.

Nao devem ser importadas funcoes, classes ou modulos criados fora do notebook.

O notebook deve seguir os seguintes principios:

- manter o codigo simples e legivel;
- evitar complexidades desnecessarias;
- reaproveitar funcoes auxiliares quando fizer sentido;
- evitar design patterns nesta etapa exploratoria;
- criar funcoes pequenas, com responsabilidade clara e docstrings objetivas;
- concentrar as funcoes auxiliares em uma unica secao do notebook.

## 4. Arquitetura Baseline a Ser Avaliada

A arquitetura inicial a ser testada sera baseada em:

- representacao textual com TF-IDF;
- Regressao Logistica;
- Random Forest;
- Gradient Boosting.

Observacao: Random Forest e Gradient Boosting devem ser tratados como comparativos exploratorios, pois podem nao ser os modelos mais adequados para matrizes TF-IDF esparsas e de alta dimensionalidade.

## 5. Ambiente e Bibliotecas

As bibliotecas necessarias devem ser instaladas inicialmente apenas no ambiente de desenvolvimento.

Bibliotecas previstas:

- pandas;
- numpy;
- scikit-learn;
- matplotlib;
- seaborn;
- wordcloud;
- nltk;
- unidecode;
- spaCy.

Tambem deve ser documentado o modelo de linguagem utilizado pelo spaCy, por exemplo `pt_core_news_sm`, caso a base esteja em portugues.

## 6. Estrutura do Notebook

### 6.1 Titulo do Notebook

Adicionar um titulo claro indicando que o notebook e uma EDA inicial com baseline para classificacao de textos hospitalares.

### 6.2 Introducao

A introducao deve conter:

- contexto do problema;
- objetivo do notebook;
- o que esperar da analise;
- link de origem da base no Kaggle (https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus?resource=download);
- observacao de que o notebook e exploratorio e nao representa ainda o pipeline final de producao.

### 6.3 Definicao do Problema

Documentar o problema que sera tratado como um problema de classificacao multiclasse, quando cada texto possui apenas uma doenca associada;

### 6.4 Imports de Bibliotecas

Todas as bibliotecas utilizadas devem ser importadas nesta secao.

Os imports devem ser organizados por blocos comentados, por exemplo:

- manipulacao de dados;
- visualizacao;
- processamento de texto;
- modelagem;
- metricas.

### 6.5 Importacao dos Dados

Importar as bases disponiveis no link do kaggle.

A secao deve conter:

- leitura do arquivo;
- visualizacao das cinco primeiras linhas com `.head()`;
- exibicao do shape da base;
- identificacao das colunas disponiveis;
- identificacao das colunas de texto e de target.

### 6.6 Entendimento Inicial da Base

Antes de realizar preprocessamento, analisar:

- quantidade de registros;
- quantidade de classes;
- distribuicao das classes;
- exemplos de textos;
- tamanho medio dos textos;
- textos vazios ou muito curtos;
- possivel desbalanceamento entre doencas/classes.

Em contexto hospitalar, exemplos textuais devem ser exibidos com cuidado para evitar exposicao de dados sensiveis.

### 6.7 Sanity Check

Verificar:

- dados faltantes;
- valores nulos;
- duplicidades;
- textos vazios;
- classes ausentes;
- duplicidades por texto;
- duplicidades por texto e classe.

Nesta etapa, nenhum tratamento deve ser aplicado. A secao deve apenas gerar um pequeno relatorio com prints estruturados.

### 6.8 Funcoes Auxiliares

Todas as funcoes utilizadas no notebook devem ser concentradas nesta secao.

As funcoes devem:

- ter nomes claros;
- executar uma responsabilidade principal;
- conter docstrings objetivas;
- documentar parametros, retorno e comportamento esperado;
- evitar complexidade desnecessaria.

Exemplos de funcoes esperadas:

- funcao para limpeza textual;
- funcao para gerar wordcloud;
- funcao para avaliar modelos;
- funcao para consolidar metricas.

### 6.9 Preprocessamento Textual

Criar colunas intermediarias para preservar o texto original e permitir comparacao antes/depois do tratamento.

A pipeline exploratoria de preprocessamento deve considerar:

1. normalizacao inicial do texto;
2. conversao para minusculas;
3. remocao de espacos extras;
4. remocao de pontuacoes;
5. remocao de numeros, se fizer sentido para o problema;
6. tokenizacao;
7. remocao de stopwords com `nltk.corpus.stopwords.words`;
8. lematizacao com spaCy;
9. reconstrucao do texto tratado.

Importante: a remocao de numeros deve ser avaliada com cuidado, pois numeros podem carregar informacao clinica relevante.

### 6.10 Exploracao dos Textos Tratados

Realizar analises comparando textos originais e tratados.

Incluir:

- distribuicao de tamanho dos textos antes e depois do preprocessamento;
- palavras mais frequentes;
- analise segmentada por classe/doenca;
- wordclouds por classificacao de doenca;
- observacoes sobre padroes textuais relevantes.

### 6.11 Separacao Treino e Teste

Antes da vetorizacao com TF-IDF, separar os dados em treino e teste.

A separacao deve:

- usar `random_state`;
- ser estratificada quando aplicavel;
- evitar vazamento de dados;
- ajustar o TF-IDF apenas nos dados de treino.

### 6.12 Modelagem Baseline

Representar os textos com TF-IDF.

Testar a base tratada com:

- Regressao Logistica;
- Random Forest;
- Gradient Boosting.

Cada modelo deve ser treinado e avaliado usando a mesma separacao de treino e teste.

### 6.13 Avaliacao dos Modelos

Avaliar os modelos com:

- recall por classe;
- precision por classe;
- f1-score por classe;
- macro recall;
- macro f1-score;
- weighted recall;
- weighted precision;
- weighted f1-score;
- matriz de confusao.

O resultado consolidado deve ser apresentado em um dataframe contendo:

- nome do modelo;
- metricas ponderadas;
- metricas macro;
- observacao sobre desempenho em classes minoritarias.

Em contexto hospitalar, a analise deve destacar especialmente o recall por classe, pois falsos negativos podem representar risco relevante.

### 6.14 Arquitetura Teorica Inicial para o MVP

Documentar uma arquitetura teorica inicial baseada nos achados do notebook.

A arquitetura deve indicar:

- entrada de texto;
- etapa de preprocessamento;
- vetorizacao;
- modelo classificador;
- saida com classe prevista e score/probabilidade;
- possibilidade de endpoint futuro para predicao;
- pontos que precisam ser refatorados antes de producao.

Essa arquitetura deve ser apresentada como proposta inicial, nao como solucao definitiva.

### 6.15 Principais Achados

Documentar:

- principais problemas encontrados nos dados;
- principais decisoes de preprocessamento;
- comportamento das classes;
- desempenho comparativo dos modelos;
- modelo baseline mais promissor;
- limitacoes da analise;
- proximos passos.

### 6.16 Relatorio de EDA em HTML

Gerar um relatorio de EDA em HTML e inserir o link ou caminho do arquivo nesta secao do notebook.

O relatorio deve conter:

1. contexto do dataset;
2. descricao do problema de classificacao de doencas;
3. etapas de preparacao dos dados utilizadas;
4. graficos e principais insights;
5. comparacao entre arquiteturas baseline;
6. modelo baseline mais promissor, com justificativa baseada em metricas;
7. sugestoes de refatoracao para uma futura estrutura de pipeline;
8. bibliotecas usadas no desenvolvimento com as versoes;
9. bibliotecas candidatas para o ambiente de producao com as versoes;
10. observacoes sobre privacidade e exposicao de dados sensiveis.

## 7. Criterios de Aceite

O notebook sera considerado adequado se:

- estiver autocontido;
- nao importar codigo local externo;
- apresentar sanity check sem tratamento imediato;
- documentar a decisao entre classificacao multiclasse;
- realizar EDA textual antes e depois do preprocessamento;
- aplicar preprocessamento textual de forma clara e reaproveitavel;
- testar os modelos definidos com TF-IDF;
- evitar vazamento de dados entre treino e teste;
- apresentar metricas por classe, macro e ponderadas;
- gerar tabela comparativa entre modelos;
- destacar limitacoes e riscos;
- gerar ou referenciar um relatorio HTML;
- propor uma arquitetura teorica inicial para evolucao do MVP.

## 8. Pontos de Atencao

- A etapa e exploratoria e nao deve ser tratada como pipeline final de producao.
- A exibicao de textos hospitalares deve respeitar privacidade e confidencialidade.
- A remocao de numeros pode eliminar informacoes clinicas importantes.
- Metricas ponderadas podem esconder baixo desempenho em classes minoritarias.
- O modelo com maior metrica geral nao necessariamente sera o mais adequado para contexto hospitalar.
- O objetivo do baseline e orientar proximas decisoes, nao encerrar a escolha da arquitetura.
