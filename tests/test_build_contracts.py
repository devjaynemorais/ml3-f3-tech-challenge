"""Reproducibility contracts for local, Docker, Airflow and CI environments."""

import ast
import sys
import tomllib
from pathlib import Path

import yaml

SPACY_MODEL = "en_core_web_sm-3.7.1"
SCISPACY_MODEL = "en_core_sci_md-0.5.4"

# Import roots whose distribution name differs from the imported module name.
# Everything else is matched by normalising ``_``/``.``/case (e.g. the
# ``pydantic_settings`` import maps to the ``pydantic-settings`` distribution).
_IMPORT_TO_DIST = {"sklearn": "scikit-learn", "yaml": "pyyaml"}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _normalise_dist(name: str) -> str:
    """Canonical distribution key: lowercase with ``_``/``.`` folded to ``-``."""
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _dist_for_import(root: str) -> str:
    """Map a top-level import name to its normalised distribution name."""
    return _IMPORT_TO_DIST.get(root, _normalise_dist(root))


def _requirement_dists(path: str) -> set[str]:
    """Distribution names declared in a pip requirements file."""
    dists = set()
    for line in _read(path).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # Strip version specifiers, extras and environment markers.
        name = line.split(";", 1)[0]
        for sep in ("<", ">", "=", "!", "~", "["):
            name = name.split(sep, 1)[0]
        dists.add(_normalise_dist(name))
    return dists


def _pyproject_runtime_dists() -> set[str]:
    """First-class runtime distributions (main + train group) from pyproject."""
    pyproject = tomllib.loads(_read("pyproject.toml"))
    poetry = pyproject["tool"]["poetry"]
    names = set(poetry["dependencies"])
    names |= set(poetry["group"]["train"]["dependencies"])
    names.discard("python")
    return {_normalise_dist(name) for name in names}


def _module_level_import_roots(package_dir: str, *, exclude: str) -> set[str]:
    """Third-party roots imported unguarded at module level under ``package_dir``.

    Imports nested in ``try``/``except`` or inside functions are skipped on
    purpose: only imports that always run at import time are hard dependencies.
    """
    roots: set[str] = set()
    for path in Path(package_dir).rglob("*.py"):
        if exclude in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    return {root for root in roots if root not in stdlib and root != package_dir}


def test_make_install_provisions_nlp_resources() -> None:
    makefile = _read("Makefile")

    assert "nltk.downloader stopwords" in makefile
    assert SPACY_MODEL in makefile
    assert SCISPACY_MODEL in makefile


def test_api_image_provisions_nlp_resources() -> None:
    dockerfile = _read("Dockerfile")

    assert "nltk.downloader" in dockerfile
    assert SPACY_MODEL in dockerfile
    assert "NLTK_DATA" in dockerfile


def test_api_receives_demo_experiment_results() -> None:
    dockerfile = _read("Dockerfile")
    compose = yaml.safe_load(_read("docker-compose.yml"))

    assert "COPY metrics/experiment_comparison.json" in dockerfile
    assert "./metrics:/app/metrics:ro" in compose["services"]["api"]["volumes"]


def test_api_receives_demo_latency_comparison() -> None:
    dockerfile = _read("Dockerfile")

    assert "COPY metrics/latency_comparison.json" in dockerfile


def test_airflow_uses_reproducible_custom_image() -> None:
    dockerfile = _read("Dockerfile.airflow")
    compose_text = _read("docker-compose.airflow.yml")
    compose = yaml.safe_load(compose_text)
    common = compose["x-airflow-common"]

    assert "apache/airflow:2.9.3-python3.11" in dockerfile
    assert SPACY_MODEL in dockerfile
    assert SCISPACY_MODEL in dockerfile
    assert "requirements-airflow.txt" in dockerfile
    assert "constraints-2.9.3/constraints-3.11.txt" in dockerfile
    assert "pip check" in dockerfile
    assert common["build"]["dockerfile"] == "Dockerfile.airflow"
    assert "_PIP_ADDITIONAL_REQUIREMENTS" not in common["environment"]


def test_airflow_build_files_belong_to_runtime_user() -> None:
    dockerfile = _read("Dockerfile.airflow")

    assert "COPY --chown=airflow:root requirements-airflow.txt" in dockerfile


def test_ci_provisions_nlp_resources() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "nltk.downloader stopwords" in workflow
    assert SPACY_MODEL in workflow
    assert SCISPACY_MODEL in workflow


def test_train_and_airflow_share_the_same_write_user() -> None:
    """Both stacks write to the shared data/models/metrics volumes, so they must
    run under the same UID — otherwise root-owned artifacts from one stack become
    unwritable by the other (the Airflow container runs as a non-root UID and
    cannot overwrite root-owned files)."""
    main = yaml.safe_load(_read("docker-compose.yml"))
    airflow = yaml.safe_load(_read("docker-compose.airflow.yml"))

    train_user = main["services"]["train"]["user"]
    airflow_user = airflow["x-airflow-common"]["user"]

    assert "AIRFLOW_UID" in train_user
    assert train_user == airflow_user


def test_airflow_requirements_cover_dag_runtime_imports() -> None:
    """Guard against requirements-airflow.txt drifting from the DAG's imports.

    The Airflow image installs its own hand-maintained requirements file rather
    than the project's pyproject. Any first-class runtime dependency that the
    DAG-executed code (everything in ``src`` except the serving layer, which the
    DAG never imports) hard-imports at module level must therefore be listed in
    requirements-airflow.txt — otherwise the DAG fails at import time in a clean
    build (as happened when xgboost, a hard import of classifier.py, was
    omitted). Transitive-only imports are ignored: only distributions that are
    also first-class pyproject runtime deps are required to be declared.
    """
    imported = {
        _dist_for_import(root)
        for root in _module_level_import_roots("src", exclude="serving")
    }
    first_class = imported & _pyproject_runtime_dists()
    declared = _requirement_dists("requirements-airflow.txt")

    missing = first_class - declared
    assert not missing, (
        f"requirements-airflow.txt missing DAG imports: {sorted(missing)}"
    )
