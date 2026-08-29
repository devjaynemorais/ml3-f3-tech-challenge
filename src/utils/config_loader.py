"""Load and validate the experiment configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
CANONICAL_LABELS = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]


class ProjectConfig(BaseModel):
    """Project metadata exposed by the API."""

    name: str
    version: str


class DataConfig(BaseModel):
    """Medical Abstracts corpus paths, schemas and canonical outputs."""

    raw_path: Path
    train_file: str
    test_file: str
    labels_file: str
    processed_path: Path
    source_text_column: str
    source_label_column: str
    labels_id_column: str
    labels_name_column: str
    text_column: str
    label_column: str
    labels: list[str]
    train_output_file: str
    validation_output_file: str
    test_output_file: str

    @field_validator("labels")
    @classmethod
    def validate_canonical_labels(cls, labels: list[str]) -> list[str]:
        """Require the exact class order shared by training and serving."""
        if labels != CANONICAL_LABELS:
            raise ValueError("labels must match the canonical order")
        return labels


class SplitConfig(BaseModel):
    """Validation split parameters for the official training data."""

    validation_size: float = Field(gt=0, lt=1)
    random_state: int


class PreprocessingConfig(BaseModel):
    """Ordered NLP preprocessing configuration."""

    steps: list[str]
    language: str
    spacy_model: str
    preserve_numbers: bool


class FeatureConfig(BaseModel):
    """TF-IDF feature parameters."""

    max_features: int = Field(gt=0)
    ngram_range: tuple[int, int]
    min_df: int = Field(gt=0)


class LogisticRegressionConfig(BaseModel):
    """Logistic Regression parameters."""

    C: float = Field(gt=0)
    max_iter: int = Field(gt=0)
    class_weight: str | None
    n_jobs: int | None
    random_state: int


class RandomForestConfig(BaseModel):
    """Random Forest parameters."""

    n_estimators: int = Field(gt=0)
    max_depth: int | None
    class_weight: str | None
    n_jobs: int | None
    random_state: int


class GradientBoostingConfig(BaseModel):
    """Gradient Boosting and feature-selection parameters."""

    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_depth: int = Field(gt=0)
    selection_k: int = Field(gt=0)
    random_state: int


class ModelConfig(BaseModel):
    """Active model Strategy and parameters for every supported option."""

    type: Literal["logistic_regression", "random_forest", "gradient_boosting"]
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig
    gradient_boosting: GradientBoostingConfig


class ArtifactConfig(BaseModel):
    """Stable paths and names for generated artifacts."""

    schema_version: int = Field(gt=0)
    model_path: Path
    pipeline_file: str
    onnx_path: Path
    onnx_file: str
    feature_pipeline_file: str
    classes_file: str
    metadata_file: str
    metrics_path: Path
    metrics_file: str


class ExperimentConfig(BaseModel):
    """Typed root configuration for the complete ML system."""

    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    features: FeatureConfig
    model: ModelConfig
    artifacts: ArtifactConfig


@lru_cache(maxsize=8)
def load_config(path: Path = CONFIG_PATH) -> ExperimentConfig:
    """Read, validate and cache a YAML experiment configuration."""
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw_config)
