# Refatoracao da arquitetura de codigo e API - Plano de implementacao

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformar o prototipo de triagem em um classificador modular das cinco categorias do Medical Abstracts TC Corpus, preservando os contratos operacionais da API, do ONNX, do Airflow, do Docker e do monitoramento.

**Architecture:** A configuracao Pydantic governa dados, preprocessamento, TF-IDF, modelo e artefatos. O pipeline sklearn encadeia um transformer composto por Strategies, TF-IDF e uma Strategy de classificador; treino, avaliacao e serving consomem os mesmos artefatos, enquanto a API FastAPI separa composicao, rotas, dependencias e schemas.

**Tech Stack:** Python 3.11, Pydantic, pandas, scikit-learn, NLTK, spaCy, Unidecode, FastAPI, ONNX Runtime, Poetry, pytest, Ruff, Docker, Airflow, Prometheus e Grafana.

**Spec:** `docs/specs/code-refactor-api-architecture.md`

## Global Constraints

- A ordem canonica das classes e `neoplasms`, `digestive system diseases`, `nervous system diseases`, `cardiovascular diseases`, `general pathological conditions`.
- O teste oficial vem integralmente de `medical_tc_test.csv`; a validacao e 10% estratificados de `medical_tc_train.csv`, com `random_state=42`.
- O pipeline padrao usa Regressao Logistica; Random Forest e Gradient Boosting continuam selecionaveis pelo YAML.
- Dependencias NLP ausentes falham com mensagem acionavel e nunca fazem download ou fallback silencioso em runtime.
- Funcoes em `src/` tem no maximo 20 linhas logicas por AST; complexidade McCabe permanece no maximo 10.
- Os endpoints `/`, `/health`, `/predict`, `/metrics` e `/docs`, as metricas Prometheus e o modo degradado sao preservados.
- Os paths publicos dos artefatos sklearn e ONNX permanecem os definidos na spec.
- Textos recebidos pela API nao sao registrados em logs.
- Nenhum repository layer, service layer generica, event bus, container de DI, MLflow ou framework novo sera introduzido.

---

### Task 1: Configuracao tipada do experimento

**Files:**
- Modify: `config/config.yaml`
- Modify: `src/utils/config_loader.py`
- Create: `tests/test_config_loader.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Produces: `load_config(path: Path = CONFIG_PATH) -> ExperimentConfig` e modelos Pydantic para `project`, `data`, `split`, `preprocessing`, `features`, `model` e `artifacts`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_load_config_returns_typed_medical_corpus_config(tmp_path: Path) -> None:
    config = load_config(write_valid_yaml(tmp_path))
    assert config.data.labels == EXPECTED_CLASSES
    assert config.split.validation_size == 0.1
    assert config.model.type == "logistic_regression"

def test_load_config_rejects_duplicate_labels(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(write_yaml_with_duplicate_labels(tmp_path))
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_config_loader.py tests/test_smoke.py -q`

Expected: FAIL because `load_config` returns a dictionary and the corpus fields do not exist.

- [ ] **Step 3: Implement typed configuration and migrate YAML**

```python
class ExperimentConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    features: FeatureConfig
    model: ModelConfig
    artifacts: ArtifactConfig

def load_config(path: Path = CONFIG_PATH) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
```

- [ ] **Step 4: Verify GREEN and lint touched files**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_config_loader.py tests/test_smoke.py -q`

Run: `.venv\\Scripts\\ruff.exe check src/utils/config_loader.py tests/test_config_loader.py tests/test_smoke.py`

- [ ] **Step 5: Commit the configuration block**

```text
Estruturar configuracao do classificador medico

refactor(config): tipar parametros de dados, NLP e modelos
```

### Task 2: Preparacao dos splits oficiais

**Files:**
- Modify: `src/data/make_dataset.py`
- Delete: `scripts/generate_synthetic_dataset.py`
- Modify: `tests/test_dataset.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `ExperimentConfig.data` e `ExperimentConfig.split`.
- Produces: `prepare_dataset(config: ExperimentConfig) -> SplitSummary` e `train.csv`, `validation.csv`, `test.csv` canonicos.

- [ ] **Step 1: Write failing corpus loader and split tests**

```python
def test_prepare_dataset_preserves_official_test_rows(tmp_path: Path) -> None:
    paths = write_medical_corpus_fixture(tmp_path)
    summary = prepare_dataset(make_config(paths))
    actual = pd.read_csv(summary.test_path)
    assert actual["text"].tolist() == OFFICIAL_TEST_TEXTS

def test_prepare_dataset_rejects_unknown_condition_id(tmp_path: Path) -> None:
    paths = write_medical_corpus_fixture(tmp_path, unknown_id=99)
    with pytest.raises(ValueError, match="unknown condition_label"):
        prepare_dataset(make_config(paths))
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_dataset.py -q`

Expected: FAIL because the current loader expects one synthetic CSV and resplits the official test.

- [ ] **Step 3: Implement validation, label mapping, stratified validation and one-time persistence**

```python
def prepare_dataset(config: ExperimentConfig) -> SplitSummary:
    raw = load_corpus(config.data)
    train, validation = split_training_data(raw.train, config.split)
    return persist_splits(train, validation, raw.test, config.data.processed_path)
```

- [ ] **Step 4: Verify GREEN and dataset invariants**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_dataset.py -q`

Run: `.venv\\Scripts\\ruff.exe check src/data/make_dataset.py tests/test_dataset.py`

- [ ] **Step 5: Commit the data block**

```text
Preparar splits oficiais do corpus medico

refactor(data): preservar teste oficial e criar validacao
```

### Task 3: Strategies e Factory de preprocessamento

**Files:**
- Modify: `src/features/text_preprocessing.py`
- Modify: `tests/test_text_preprocessing.py`
- Modify: `pyproject.toml`
- Modify: `poetry.lock`

**Interfaces:**
- Consumes: `PreprocessingConfig`.
- Produces: `PreprocessingStrategy`, `PreprocessingFactory.create(step_name, config)` e `TextPreprocessor(BaseEstimator, TransformerMixin)`.

- [ ] **Step 1: Write failing isolated Strategy and composition tests**

```python
def test_preprocessing_pipeline_preserves_numbers_and_removes_noise(resources) -> None:
    transformer = TextPreprocessor(TEST_CONFIG)
    assert transformer.transform(["  PATIENT'S dose: 220 mg.  "])[0] == "patient dose 220 mg"

def test_factory_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        PreprocessingFactory.create("unknown", TEST_CONFIG)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_text_preprocessing.py -q`

Expected: FAIL because Protocol, Factory, Strategies and composed transformer are absent.

- [ ] **Step 3: Implement four Strategies and actionable resource failures**

```python
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, texts: Iterable[str], y: object = None) -> TextPreprocessor:
        return self

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [self._transform_one(text) for text in texts]
```

- [ ] **Step 4: Update and lock main NLP dependencies**

Run: `.venv\\Scripts\\python.exe -m poetry lock`

- [ ] **Step 5: Verify GREEN**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_text_preprocessing.py -q`

Run: `.venv\\Scripts\\ruff.exe check src/features/text_preprocessing.py tests/test_text_preprocessing.py`

- [ ] **Step 6: Commit the preprocessing block**

```text
Modularizar preprocessamento textual

refactor(features): aplicar factory e strategies de NLP
```

### Task 4: Strategies e Factory de modelos

**Files:**
- Modify: `src/models/classifier.py`
- Modify: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `ExperimentConfig.preprocessing`, `ExperimentConfig.features` e `ExperimentConfig.model`.
- Produces: `ModelStrategy`, `ModelFactory.create(model_type)` e `build_pipeline(config) -> Pipeline`.

- [ ] **Step 1: Write failing factory, parameter and five-class probability tests**

```python
@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_each_model_trains_and_returns_five_probabilities(model_type: str) -> None:
    pipeline = build_pipeline(make_config(model_type))
    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    assert pipeline.predict_proba(["sample abstract"]).shape == (1, 5)

def test_gradient_boosting_owns_chi2_and_dense_conversion() -> None:
    names = [name for name, _ in ModelFactory.create("gradient_boosting").build_steps(MODEL_CONFIG)]
    assert names == ["feature_selection", "to_dense", "classifier"]
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_classifier.py -q`

Expected: FAIL because the model factory and Gradient Boosting branch do not exist.

- [ ] **Step 3: Implement model Strategies and common pipeline composition**

```python
def build_pipeline(config: ExperimentConfig) -> Pipeline:
    prefix = [("preprocessor", TextPreprocessor(config.preprocessing)), ("tfidf", build_tfidf(config.features))]
    suffix = ModelFactory.create(config.model.type).build_steps(config.model)
    return Pipeline([*prefix, *suffix])
```

- [ ] **Step 4: Verify GREEN**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_classifier.py -q`

Run: `.venv\\Scripts\\ruff.exe check src/models/classifier.py tests/test_classifier.py`

- [ ] **Step 5: Commit the model block**

```text
Organizar modelos com factory e strategy

refactor(models): isolar construcao dos classificadores
```

### Task 5: Treino, avaliacao e registro de artefatos

**Files:**
- Modify: `src/training/trainer.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `src/models/registry.py`
- Create: `tests/test_training.py`
- Create: `tests/test_evaluation.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Consumes: somente CSVs processados e `ExperimentConfig`.
- Produces: pipeline joblib, metadata schema v2 e `metrics/eval_metrics.json` com secoes `validation` e `test`.

- [ ] **Step 1: Write failing no-resplit, metrics and metadata tests**

```python
def test_train_uses_only_processed_training_split(tmp_path: Path) -> None:
    pipeline, metadata = train_from_processed(make_config(tmp_path))
    assert metadata.n_train_samples == len(TRAIN_ROWS)
    assert list(pipeline.classes_) == EXPECTED_CLASSES_SORTED_BY_SKLEARN

def test_evaluate_reports_validation_and_test_without_refit(fitted_pipeline) -> None:
    metrics = evaluate_splits(fitted_pipeline, VALIDATION_DF, TEST_DF, CONFIG)
    assert set(metrics) == {"validation", "test"}
    assert metrics["test"]["labels"] == EXPECTED_CLASSES
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_training.py tests/test_evaluation.py tests/test_registry.py -q`

Expected: FAIL because training/evaluation resplit raw data and metadata lacks the versioned schema.

- [ ] **Step 3: Implement processed-split training, complete metrics and versioned metadata**

```python
def evaluate_splits(pipeline: Pipeline, validation: DataFrame, test: DataFrame, config: ExperimentConfig) -> dict[str, dict[str, object]]:
    return {
        "validation": evaluate_split(pipeline, validation, config.data.labels),
        "test": evaluate_split(pipeline, test, config.data.labels),
    }
```

- [ ] **Step 4: Verify GREEN and round trip**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_training.py tests/test_evaluation.py tests/test_registry.py -q`

- [ ] **Step 5: Commit the training block**

```text
Refatorar treino e avaliacao do classificador

refactor(pipeline): reutilizar splits e ampliar metricas
```

### Task 6: Exportacao ONNX e predictors validados

**Files:**
- Modify: `src/optimization/export_onnx.py`
- Modify: `src/serving/model_loader.py`
- Modify: `scripts/measure_latency.py`
- Modify: `tests/test_model_loader.py`
- Create: `tests/test_onnx.py`

**Interfaces:**
- Produces: prefixo de features joblib, classificador ONNX, classes JSON e predictors que validam exatamente as cinco classes.

- [ ] **Step 1: Write failing prefix, old-artifact and parity tests**

```python
def test_old_three_class_artifact_is_rejected(trained_three_class_pipeline) -> None:
    with pytest.raises(ValueError, match="artifact classes"):
        SklearnPredictor(trained_three_class_pipeline, EXPECTED_CLASSES)

def test_sklearn_and_onnx_predictions_are_compatible(exported_predictors) -> None:
    sk_label, sk_scores = exported_predictors.sklearn.predict(SAMPLE_TEXT)
    onnx_label, onnx_scores = exported_predictors.onnx.predict(SAMPLE_TEXT)
    assert onnx_label == sk_label
    np.testing.assert_allclose(list(onnx_scores.values()), list(sk_scores.values()), rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_model_loader.py tests/test_onnx.py -q`

Expected: FAIL because class validation and generalized feature prefix export are absent.

- [ ] **Step 3: Implement classifier split, feature prefix persistence and shared predictor benchmark**

```python
def split_feature_prefix(pipeline: Pipeline) -> tuple[Pipeline, BaseEstimator]:
    classifier_name = pipeline.steps[-1][0]
    return Pipeline(pipeline.steps[:-1]), pipeline.named_steps[classifier_name]
```

- [ ] **Step 4: Verify GREEN and ONNX parity**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_model_loader.py tests/test_onnx.py -q`

- [ ] **Step 5: Commit the ONNX block**

```text
Adaptar inferencia e exportacao ONNX

refactor(onnx): persistir pipeline completo de features
```

### Task 7: Composicao modular da API FastAPI

**Files:**
- Modify: `src/serving/api.py`
- Create: `src/serving/routes.py`
- Create: `src/serving/dependencies.py`
- Modify: `src/serving/schemas.py`
- Modify: `src/serving/metrics.py`
- Modify: `tests/test_serving.py`

**Interfaces:**
- Produces: `create_app(predictor_loader: Callable[[], TriagePredictor] = load_predictor) -> FastAPI`, `app`, rotas publicas e dependencia baseada em `request.app.state.predictor`.

- [ ] **Step 1: Write failing factory, validation, degraded and error-handling tests**

```python
def test_blank_text_returns_422(app_with_predictor) -> None:
    response = TestClient(app_with_predictor).post("/predict", json={"text": "   "})
    assert response.status_code == 422

def test_unexpected_inference_failure_returns_500_without_logging_text(caplog) -> None:
    response = TestClient(app_with_failing_predictor).post("/predict", json={"text": SECRET_TEXT})
    assert response.status_code == 500
    assert SECRET_TEXT not in caplog.text
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_serving.py -q`

Expected: FAIL because the current module uses a global dictionary and whitespace-only text passes validation.

- [ ] **Step 3: Implement app factory, lifespan state, routes, dependencies and five-class schemas**

```python
def create_app(predictor_loader: PredictorLoader = load_predictor) -> FastAPI:
    application = FastAPI(lifespan=create_lifespan(predictor_loader), **OPENAPI_METADATA)
    application.add_middleware(PrometheusMiddleware)
    application.include_router(router)
    return application
```

- [ ] **Step 4: Verify GREEN including Swagger and Prometheus**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_serving.py tests/test_metrics.py -q`

- [ ] **Step 5: Commit the API block**

```text
Estruturar API de classificacao medica

refactor(api): separar composicao, rotas e dependencias
```

### Task 8: Recursos NLP, Docker, Airflow, CI e documentacao

**Files:**
- Modify: `Makefile`
- Modify: `Dockerfile`
- Create: `Dockerfile.airflow`
- Modify: `docker-compose.airflow.yml`
- Modify: `.github/workflows/ci.yml`
- Create: `tests/test_source_function_length.py`
- Modify: `airflow/dags/triage_training_dag.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/model_card.md`
- Modify: `postman/Triage-API.postman_collection.json`
- Modify: `monitoring/grafana/dashboards/triage-api-overview.json`
- Modify: `AGENTS.md`
- Create: `tests/test_airflow_dag.py`

**Interfaces:**
- Consumes: os comandos e contratos implementados nas Tasks 1-7.
- Produces: instalacao reproduzivel de NLTK/spaCy, Airflow customizado, teste AST e documentacao coerente com o corpus de abstracts em ingles.

- [ ] **Step 1: Write failing AST and DAG tests**

```python
def test_src_functions_have_at_most_twenty_logical_lines() -> None:
    violations = find_function_length_violations(Path("src"), maximum=20)
    assert violations == []

def test_ingest_task_prepares_corpus_without_synthetic_generation(monkeypatch) -> None:
    assert run_ingest_task(monkeypatch).endswith("data/processed/train.csv")
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_source_function_length.py tests/test_airflow_dag.py -q`

Expected: FAIL on current long functions and synthetic Airflow ingestion.

- [ ] **Step 3: Update install/build resources, Airflow image, CI and documentation**

```dockerfile
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords \
    && python -m spacy download en_core_web_sm==3.8.0
```

- [ ] **Step 4: Verify lint, unit tests and static infrastructure configuration**

Run: `make lint`

Run: `make test`

Run: `docker compose config`

Run: `docker compose -f docker-compose.airflow.yml config`

- [ ] **Step 5: Commit infrastructure and documentation in coherent blocks**

```text
Atualizar ambientes do pipeline NLP

build(nlp): instalar recursos reproduziveis nas imagens
```

```text
Documentar arquitetura refatorada

docs(architecture): alinhar corpus, modelos e API
```

### Task 9: End-to-end verification and local API exercise

**Files:**
- Create locally and keep ignored: `data/raw/medical_tc_train.csv`, `data/raw/medical_tc_test.csv`, `data/raw/medical_tc_labels.csv`, processed splits, model artifacts and metrics.
- Create locally and keep ignored when useful: `tests/requests/*.json` or equivalent temporary request payloads.

**Interfaces:**
- Consumes: every public command and endpoint from the spec.
- Produces: fresh command evidence, API response evidence and final commit SHAs.

- [ ] **Step 1: Install and verify NLP resources**

Run: `make install`

Run: `.venv\\Scripts\\python.exe -c "from nltk.corpus import stopwords; import spacy; print(len(stopwords.words('english'))); print(spacy.load('en_core_web_sm').meta['version'])"`

- [ ] **Step 2: Run the complete ML pipeline**

Run: `make dataset`

Run: `make train`

Run: `make evaluate`

Run: `make export-onnx`

Run: `make benchmark-latency`

- [ ] **Step 3: Run quality and container verification**

Run: `make lint`

Run: `make test`

Run: `make compose-build`

- [ ] **Step 4: Start the local API and exercise every GET endpoint**

Run: `make api`

Requests: `GET /`, `GET /health`, `GET /metrics`, `GET /docs`.

Expected: HTTP 200 for all, `/health` reports the configured backend and `/metrics` exposes the preserved metric names.

- [ ] **Step 5: Exercise five representative JSON prediction requests**

```json
{"text":"The biopsy showed invasive carcinoma with metastatic involvement."}
{"text":"Endoscopy identified chronic inflammation of the gastric mucosa."}
{"text":"The patient developed progressive tremor and impaired motor coordination."}
{"text":"Electrocardiography demonstrated atrial fibrillation and ventricular dysfunction."}
{"text":"Laboratory evaluation showed fever, fatigue, and a systemic inflammatory response."}
```

Expected: HTTP 200, one canonical label, five scores, approximate sum 1 and the configured backend for every response.

- [ ] **Step 6: Build and inspect Airflow when the local Docker runtime permits it**

Run: `make airflow-up`

Expected: DAG `triage_training` is visible, paused at creation, and ordered as `ingest_data -> train_model -> evaluate_model -> export_onnx`.

- [ ] **Step 7: Perform read-only code review and resolve every Critical or Important finding**

Run: `git diff <base-sha>..<head-sha>` plus the complete test suite after fixes.

- [ ] **Step 8: Confirm semantic GitHub commits and clean status**

Run: `git status --short --branch`

Expected: only ignored runtime artifacts remain outside version control and every implementation block has a semantic commit.
