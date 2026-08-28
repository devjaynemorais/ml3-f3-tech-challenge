"""Shared validation for classifier artifact classes."""

from collections.abc import Sequence


def validate_artifact_classes(
    actual: Sequence[str], expected: Sequence[str]
) -> list[str]:
    """Require one occurrence of every configured class in an artifact."""
    actual_classes = list(actual)
    expected_classes = list(expected)
    valid = len(actual_classes) == len(set(actual_classes)) == len(expected_classes)
    if not valid or set(actual_classes) != set(expected_classes):
        raise ValueError(
            f"artifact classes {actual_classes!r} do not match {expected_classes!r}"
        )
    return actual_classes
