from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config_loader import load_config

EXPECTED_CLASSES = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]


def test_load_config_returns_typed_medical_corpus_config() -> None:
    config = load_config()

    assert config.data.labels == EXPECTED_CLASSES
    assert config.data.train_file == "medical_tc_train.csv"
    assert config.data.test_file == "medical_tc_test.csv"
    assert config.data.labels_file == "medical_tc_labels.csv"
    assert config.split.validation_size == 0.1
    assert config.model.type == "complement_nb"


def test_load_config_rejects_duplicate_labels(tmp_path: Path) -> None:
    source_path = Path("config/config.yaml")
    raw_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw_config["data"]["labels"] = [EXPECTED_CLASSES[0]] * 5
    invalid_path = tmp_path / "invalid-config.yaml"
    invalid_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    with pytest.raises(ValidationError, match="canonical order"):
        load_config(invalid_path)
