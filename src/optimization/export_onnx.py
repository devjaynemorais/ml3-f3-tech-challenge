"""Export the fitted classifier to ONNX and persist its feature prefix."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from src.models.artifact_contract import validate_artifact_classes
from src.models.registry import load_pipeline
from src.utils.config_loader import ArtifactConfig, load_config

if TYPE_CHECKING:
    from onnx import ModelProto
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnnxArtifactPaths:
    """Paths written by one ONNX export."""

    model: Path
    feature_pipeline: Path
    classes: Path


def split_feature_prefix(pipeline: Pipeline) -> tuple[Pipeline, BaseEstimator]:
    """Separate all feature steps from the final classifier."""
    step_name, classifier = pipeline.steps[-1]
    if step_name != "classifier":
        raise ValueError("the final pipeline step must be named 'classifier'")
    return Pipeline(pipeline.steps[:-1]), classifier


def export_classifier_to_onnx(classifier: BaseEstimator) -> ModelProto:
    """Convert one fitted classifier with dense probability output."""
    n_features = int(classifier.n_features_in_)
    initial_type = [("input", FloatTensorType([None, n_features]))]
    return convert_sklearn(
        classifier,
        initial_types=initial_type,
        options={id(classifier): {"zipmap": False}},
    )


def _artifact_paths(path: Path, config: ArtifactConfig) -> OnnxArtifactPaths:
    """Resolve the stable ONNX artifact paths."""
    return OnnxArtifactPaths(
        path / config.onnx_file,
        path / config.feature_pipeline_file,
        path / config.classes_file,
    )


def export_pipeline_artifacts(
    pipeline: Pipeline,
    onnx_path: Path,
    config: ArtifactConfig,
    expected_classes: list[str],
) -> OnnxArtifactPaths:
    """Persist feature prefix, ONNX classifier and probability class order."""
    prefix, classifier = split_feature_prefix(pipeline)
    classes = validate_artifact_classes(list(classifier.classes_), expected_classes)
    onnx_model = export_classifier_to_onnx(classifier)
    onnx_path.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(onnx_path, config)
    paths.model.write_bytes(onnx_model.SerializeToString())
    joblib.dump(prefix, paths.feature_pipeline)
    paths.classes.write_text(json.dumps(classes), encoding="utf-8")
    return paths


def main() -> None:
    """Load the fitted pipeline and write every ONNX serving artifact."""
    config = load_config()
    pipeline = load_pipeline(
        config.artifacts.model_path, config.artifacts.pipeline_file
    )
    paths = export_pipeline_artifacts(
        pipeline, config.artifacts.onnx_path, config.artifacts, config.data.labels
    )
    logger.info("ONNX classifier exported to %s", paths.model)


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
