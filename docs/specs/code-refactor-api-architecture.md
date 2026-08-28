# SPEC — Refatoração da arquitetura de código e API

## 1. Objetivo e decisões

Refatorar o pipeline explorado no Notebook 01 para uma arquitetura modular de
produção, mantendo Python 3.11, scikit-learn, FastAPI, Poetry, ONNX, Airflow e
os contratos operacionais existentes.

Decisões consolidadas:

- Domínio: classificação multiclasse das cinco categorias do Medical
  Abstracts TC Corpus.
- Ordem canônica: `neoplasms`, `digestive system diseases`,
  `nervous system diseases`, `cardiovascular diseases` e
  `general pathological conditions`.
- Dados: preservar `medical_tc_test.csv` como teste oficial; retirar 10%
  estratificados de `medical_tc_train.csv` para validação.
- Modelo: uma Strategy ativa por execução, escolhida no YAML; Regressão
  Logística como padrão.
- Preprocessamento: composição configurável de Strategies, criada por Factory.
- Dependências NLP ausentes: falha explícita, sem download ou fallback
  silencioso em runtime.
- Funções de `src/`: máximo de 20 linhas lógicas, verificadas por AST.
- Patterns adicionais, repository layer, service layer genérica, event bus e
  container de DI não serão introduzidos.

Fora do escopo:

- Classificação de urgência `normal`/`atencao`/`urgente`.
- Seleção automática do melhor modelo.
- Classificação multilabel, diagnóstico clínico, batch API, MLflow ou novo
  framework.
- Alteração dos endpoints, métricas Prometheus, stages Docker ou ordem da DAG.

## 2. Arquitetura e interfaces

```text
medical_tc_{train,test,labels}.csv
          │
          ▼
validação + mapeamento de labels + split de validação
          │
          ├── data/processed/train.csv
          ├── data/processed/validation.csv
          └── data/processed/test.csv
                     │
                     ▼
Strategies de preprocessamento
  → TF-IDF
  → Strategy do modelo
  → pipeline sklearn
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
triage_pipeline.joblib     prefixo de features
                               + classificador ONNX
          │                     │
          └──────── TriagePredictor ────────┐
                                             ▼
                            FastAPI /predict → label + scores + backend
```

### 2.1 Configuração e dados

Criar configuração tipada com Pydantic:

```python
def load_config(path: Path = Path("config/config.yaml")) -> ExperimentConfig
```

O YAML passará a definir:

- Os três arquivos do corpus, nomes das colunas originais e colunas canônicas
  `text`/`label`.
- A lista ordenada das cinco classes.
- `validation_size: 0.1` e `random_state: 42`.
- Ordem das etapas de preprocessamento, idioma, modelo spaCy e preservação de
  números.
- TF-IDF com `max_features=5000`, `ngram_range=[1, 2]` e `min_df=2`.
- Modelo ativo e parâmetros de cada alternativa.
- Paths e nomes dos artefatos existentes.

A preparação dos dados deverá:

- Validar existência dos três CSVs e seus schemas.
- Validar unicidade do mapeamento `condition_label → condition_name`.
- Rejeitar IDs ou classes desconhecidos.
- Renomear as colunas internamente para `text` e `label`.
- Remover apenas registros sem texto ou label, registrando quantidades.
- Criar validação estratificada somente a partir do treino oficial.
- Manter o teste oficial integral e sem reutilização no treino.
- Persistir os três splits processados uma única vez; treino e avaliação não
  poderão refazer o split.

`make dataset` passará a preparar o corpus real. O gerador sintético das três
urgências será removido do fluxo e dos testes.

### 2.2 Strategy e Factory de preprocessamento

Contrato mínimo:

```python
class PreprocessingStrategy(Protocol):
    def transform(self, text: str) -> str: ...


class PreprocessingFactory:
    @classmethod
    def create(
        cls,
        step_name: str,
        config: PreprocessingConfig,
    ) -> PreprocessingStrategy: ...
```

Strategies obrigatórias, nesta ordem padrão:

1. Normalização Unicode, lowercase e espaços.
2. Remoção de pontuação, preservando letras e números.
3. Tokenização e lematização com `en_core_web_sm`.
4. Remoção de stopwords NLTK sobre os lemas, descartando tokens de um
   caractere.

`TextPreprocessor` será o transformer sklearn responsável apenas por executar
a lista. Nomes inválidos, stopwords ausentes ou modelo spaCy indisponível
deverão produzir erros acionáveis. A API deverá usar exatamente o preprocessor
persistido no artefato, evitando divergência entre treino e inferência.

Adicionar `nltk>=3.10,<4`, `spacy>=3.8,<3.9` e `Unidecode>=1.4,<2` às
dependências principais e versionar o lock. Instalar `en_core_web_sm 3.8.0`,
versão validada no relatório exploratório. Os recursos serão instalados em
`make install` e durante o build das imagens, nunca na primeira predição.

### 2.3 Strategy e Factory de modelos

Contrato mínimo:

```python
class ModelStrategy(Protocol):
    def build_steps(
        self,
        config: ModelConfig,
    ) -> list[tuple[str, BaseEstimator]]: ...


class ModelFactory:
    @classmethod
    def create(cls, model_type: str) -> ModelStrategy: ...
```

Implementações:

- Regressão Logística: `C=1.0`, `max_iter=1000`,
  `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`.
- Random Forest: `n_estimators=160`, `max_depth=None`,
  `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`.
- Gradient Boosting: `SelectKBest(chi2, k=min(800, n_features))`, conversão
  densa e `GradientBoostingClassifier(n_estimators=80, learning_rate=0.08,
  max_depth=3, random_state=42)`.

Somente a Strategy de Gradient Boosting conhecerá seleção `chi²` e conversão
densa. O builder comum apenas concatenará:

```text
preprocessor → tfidf → model_strategy.build_steps()
```

### 2.4 Treino, avaliação e artefatos

- Treino consumirá exclusivamente `data/processed/train.csv`.
- Avaliação consumirá validação e teste, sem alterar ou reajustar o pipeline.
- `eval_metrics.json` terá seções `validation` e `test`.
- Cada seção conterá accuracy, precision/recall/F1 por classe, métricas macro e
  weighted, recall médio das duas classes minoritárias, labels e matriz de
  confusão.
- Metadata registrará versão do schema de artefato, modelo, parâmetros,
  Strategies de preprocessamento, classes e quantidade de amostras.
- O loader validará se as classes do artefato correspondem exatamente às cinco
  classes configuradas; artefatos antigos de urgência deixarão a API degradada
  e exigirão retreino.

Preservar:

- `models/artifacts/triage_pipeline.joblib`;
- `models/artifacts/model_metadata.json`;
- `models/onnx/triage_classifier.onnx`;
- `models/onnx/vectorizer.joblib`;
- `models/onnx/classes.json`.

No backend ONNX, `vectorizer.joblib` passará a armazenar todo o prefixo anterior
ao classificador: preprocessamento, TF-IDF e, quando necessário,
seleção/conversão. Apenas o classificador será convertido. O número de features
será obtido de `classifier.n_features_in_`.

### 2.5 Estrutura da API

Manter `src.serving.api:app` como entrypoint, dividindo responsabilidades:

- `api.py`: `create_app()`, lifespan, middleware e composição.
- `routes.py`: `/`, `/health`, `/predict` e `/metrics`.
- `dependencies.py`: acesso ao predictor presente em `app.state`.
- `schemas.py`: request, response, health e enum das cinco classes.
- `model_loader.py`: protocolo e backends sklearn/ONNX.

Contrato público:

| Endpoint | Comportamento |
|---|---|
| `GET /` | Nome e versão do classificador de textos médicos |
| `GET /health` | `ok` com backend ou `degraded` sem artefato válido |
| `POST /predict` | Recebe `{"text": "..."}` e retorna `label`, `scores`, `backend` |
| `GET /metrics` | Mantém métricas Prometheus existentes |
| `GET /docs` | OpenAPI/Swagger do FastAPI |

Regras:

- Texto vazio ou composto apenas por espaços: HTTP 422.
- Predictor ausente: HTTP 503.
- Falha de carregamento no startup: API sobe degradada.
- Falha inesperada de inferência: HTTP 500 com log técnico.
- Textos recebidos nunca serão incluídos em logs.
- `scores` conterá as cinco classes e probabilidades com soma aproximada igual
  a 1.
- O estado global em dicionário será substituído por `app.state.predictor`.

## 3. Etapas de implementação

Todas as etapas seguirão TDD: escrever teste falhando, confirmar a falha,
implementar o mínimo, executar testes, refatorar e criar commit semântico.

### 3.1 Caracterização e configuração

- Preservar em testes os endpoints, modo degradado, métricas e paths de
  artefato.
- Criar modelos Pydantic da configuração e migrar `load_config`.
- Atualizar YAML para o corpus e parâmetros aprovados.
- Commit sugerido:

  ```text
  Estruturar configuração do classificador médico

  refactor(config): tipar parâmetros de dados, NLP e modelos
  ```

### 3.2 Pipeline de dados

- Separar loader do Medical Abstracts, validação, split e persistência.
- Atualizar `make dataset` e remover o gerador de urgências.
- Testar schemas inválidos, labels desconhecidas, determinismo e integridade do
  teste oficial.
- Commit sugerido:

  ```text
  Preparar splits oficiais do corpus médico

  refactor(data): preservar teste oficial e criar validação
  ```

### 3.3 Preprocessamento

- Criar Protocol, quatro Strategies, Factory e transformer composto.
- Testar ordem, lowercase, pontuação, espaços, lematização, stopwords,
  preservação de `220` e erros de recursos.
- Commit sugerido:

  ```text
  Modularizar preprocessamento textual

  refactor(features): aplicar factory e strategies de NLP
  ```

### 3.4 Modelos

- Criar Protocol, Strategies e Factory dos três modelos.
- Adaptar o builder do pipeline.
- Testar tipos válidos/inválidos, parâmetros e branch esparso→denso do GB.
- Commit sugerido:

  ```text
  Organizar modelos com factory e strategy

  refactor(models): isolar construção dos classificadores
  ```

### 3.5 Treino, avaliação e registro

- Remover split duplicado de treino e avaliação.
- Produzir métricas completas de validação/teste e metadata versionada.
- Testar round-trip joblib, cinco classes e ausência de vazamento.
- Commit sugerido:

  ```text
  Refatorar treino e avaliação do classificador

  refactor(pipeline): reutilizar splits e ampliar métricas
  ```

### 3.6 ONNX e inferência

- Generalizar a separação entre prefixo de features e classificador.
- Validar classes e artefatos nos dois backends.
- Atualizar benchmark para texto sintético em inglês e reutilizar
  `TriagePredictor`, eliminando lógica ONNX duplicada do script.
- Testar paridade de label e scores com tolerância `rtol=1e-5`, `atol=1e-6`.
- Commit sugerido:

  ```text
  Adaptar inferência e exportação ONNX

  refactor(onnx): persistir pipeline completo de features
  ```

### 3.7 API

- Introduzir app factory, rotas, dependências e schemas.
- Substituir estado global e documentar as cinco classes.
- Testar 200, 422, 503, health degradado, Swagger e métricas.
- Commit sugerido:

  ```text
  Estruturar API de classificação médica

  refactor(api): separar composição, rotas e dependências
  ```

### 3.8 Docker, Airflow e CI

- Instalar recursos NLP nas imagens de treino/API.
- Criar `Dockerfile.airflow` a partir da imagem Airflow atual, instalar nele as
  dependências e os recursos NLP e configurar `docker-compose.airflow.yml`
  para construir essa imagem; remover `_PIP_ADDITIONAL_REQUIREMENTS` do
  startup.
- Manter `ingest_data → train_model → evaluate_model → export_onnx`.
- Estender CI para o teste AST e testes de integração que não dependam dos CSVs
  locais.
- Commit sugerido:

  ```text
  Atualizar ambientes do pipeline NLP

  build(nlp): instalar recursos reproduzíveis nas imagens
  ```

### 3.9 Documentação e reconciliação

- Atualizar README, arquitetura, model card, exemplos de API e descrições do
  dashboard.
- Atualizar somente os trechos conflitantes do `AGENTS.md`, preservando o
  restante do arquivo não commitado.
- Declarar que a base é de abstracts públicos em inglês e não representa
  triagem hospitalar real nem validação clínica.
- Commit sugerido:

  ```text
  Documentar arquitetura refatorada

  docs(architecture): alinhar corpus, modelos e API
  ```

## 4. Testes e critérios de aceite

Testes obrigatórios:

- Factory rejeita nomes de preprocessamento e modelos desconhecidos.
- Todas as Strategies cumprem os Protocols e funcionam isoladamente.
- Números clínicos permanecem no texto; pontuação e stopwords são removidas.
- TF-IDF é ajustado somente no treino.
- Teste oficial não participa do fit nem do split de validação.
- LR, RF e GB treinam e retornam probabilidades para cinco classes.
- GB recebe matriz densa somente após `chi²`.
- Artefato sklearn preserva resultados após save/load.
- Artefato antigo com três classes é rejeitado.
- Backends sklearn e ONNX retornam mesma label e scores numericamente
  compatíveis.
- API preserva todos os endpoints, modo degradado e métricas.
- Nenhuma fixture, log ou documentação expõe texto clínico real completo.
- Teste AST falha quando uma função de `src/` exceder 20 linhas lógicas.
- Ruff mantém complexidade McCabe máxima 10.

Validação final:

```text
make lint
make test
make dataset
make train
make evaluate
make export-onnx
make benchmark-latency
make compose-build
make compose-up
make airflow-up
```

Confirmar manualmente `/health`, `/predict`, `/metrics`, `/docs`, dashboard e
execução ordenada da DAG. Dados, modelos e métricas gerados permanecerão fora
do Git.

## 5. Revisão contra o AGENTS.md

- Preserva FastAPI, REST, Docker, CI/CD, Airflow, Prometheus, Grafana e ONNX.
- Preserva configuração de experimento no YAML e runtime no `.env`.
- Preserva paths principais de artefatos e contrato `TriagePredictor`.
- Preserva endpoints e comportamento degradado.
- Preserva as métricas Prometheus e stages `builder`, `train`, `api`.
- Mantém a DAG pausada e sem duplicação de lógica.
- Atualiza os contratos antigos de três urgências conforme a decisão explícita
  de adotar as cinco classes definidas no objetivo principal.
- Mantém o notebook autocontido; nenhum código de produção será importado por
  ele.
- Mantém comunicação clínica responsável, revisão humana e ausência de claims
  diagnósticos.
- Não introduz patterns além de Factory e Strategy; a composição das etapas
  usa apenas o pipeline sklearn já necessário.

Antes de considerar a spec pronta para implementação, executar revisão de
placeholders, contradições, interfaces, paths, cobertura de requisitos e
aderência a DRY, SOLID e KISS.

