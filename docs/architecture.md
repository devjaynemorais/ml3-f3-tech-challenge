# Arquitetura

## Fluxo de dados e inferência

```text
medical_tc_{train,test,labels}.csv
          |
          v
validação + mapeamento + split estratificado de validação
          |
          +--> data/processed/train.csv
          +--> data/processed/validation.csv
          +--> data/processed/test.csv (teste oficial integral)
                         |
                         v
Strategies de preprocessing -> TF-IDF -> Strategy do modelo
                         |
             +-----------+-----------+
             v                       v
triage_pipeline.joblib       prefixo de features + ONNX
             |                       |
             +------- TriagePredictor+
                         |
                         v
FastAPI /predict -> label + cinco scores + backend
```

O domínio é a classificação de abstracts públicos em inglês nas categorias
`neoplasms`, `digestive system diseases`, `nervous system diseases`,
`cardiovascular diseases` e `general pathological conditions`. Esse corpus não
representa fluxo hospitalar real nem constitui validação clínica.

## Configuração e separação de responsabilidades

`config/config.yaml` concentra paths, classes, split, preprocessing, TF-IDF,
modelos e nomes de artefatos. `src/utils/config_loader.py` valida o YAML com
Pydantic antes de qualquer etapa.

O pipeline de dados valida os três CSVs, canonicaliza as colunas como `text` e
`label`, cria validação apenas do treino oficial e persiste os splits uma vez.
Treino e avaliação apenas leem os arquivos processados; o teste oficial não é
usado no fit.

Preprocessing e modelo usam Factory e Strategy. O builder comum concatena o
preprocessor configurado, TF-IDF e os passos do modelo ativo. Regressão
Logística é o padrão; Random Forest e Gradient Boosting são alternativas
explícitas. Somente a Strategy de Gradient Boosting aplica chi-quadrado e
conversão densa.

## Serving real-time

`src/serving/api.py` compõe a aplicação e seu lifespan. Rotas, dependência do
predictor, schemas, métricas e backends ficam em módulos separados. Cada
instância armazena seu predictor em `app.state`, evitando estado global
compartilhado.

A API pode iniciar sem modelo: `/health` retorna `degraded` e `/predict` retorna
503. Isso separa disponibilidade da infraestrutura da promoção do artefato. Um
artefato com classes diferentes das cinco classes canônicas é rejeitado.

## Docker

O `Dockerfile` preserva os stages:

- `builder`: instala dependências, stopwords NLTK e `en_core_web_sm 3.8.0`;
- `train`: executa preparação, treino, avaliação, exportação e benchmark;
- `api`: contém o runtime de serving e reutiliza o ambiente do builder.

Os recursos NLP são instalados no build, nunca durante a primeira predição.
`Dockerfile.airflow` estende a imagem oficial do Airflow com as mesmas
dependências e recursos. O Compose do Airflow fica separado da stack de serving.

## ONNX

O backend ONNX mantém em joblib todo o prefixo anterior ao classificador:
preprocessing, TF-IDF e eventuais seleção/conversão de features. Apenas o
classificador final é convertido, com número de features obtido de
`classifier.n_features_in_`.

Essa fronteira evita introduzir um runtime adicional de tokenização e preserva
o preprocessing treinado. O benchmark chama o mesmo contrato
`TriagePredictor` usado pela API e compara sklearn/ONNX com textos sintéticos.

## Airflow e CI/CD

A DAG `triage_training` é manual (`schedule=None`) e mantém a sequência:

```text
ingest_data -> train_model -> evaluate_model -> export_onnx
```

Cada task chama um entrypoint de `src/`; nenhuma lógica de ML é copiada para a
DAG. O CI instala os recursos NLP, executa Ruff, pytest com cobertura — incluindo
o limite AST de 20 linhas lógicas por função — e constrói o stage `api`.

## Observabilidade

- `http_requests_total{method,path,status_code}` mede volume e erros;
- `http_request_duration_seconds{method,path}` alimenta p50/p95;
- `triage_predictions_total{predicted_label}` acompanha a distribuição das
  cinco categorias médicas.

O dashboard Grafana apresenta volume, taxa de erro, latência, requisições por
rota e distribuição das predições. Mudança de distribuição é um sinal para
investigação, não evidência clínica isolada.

## Deploy em nuvem

O desenho recomendado no README usa ECR, ECS Fargate e Application Load
Balancer, com `/health` como health check. CloudWatch e serviços gerenciados de
Prometheus podem receber a mesma telemetria. Amazon MWAA pode executar a DAG de
retreino sem alterar a lógica do pipeline.
