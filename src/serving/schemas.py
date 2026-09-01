"""Pydantic request and response contracts for the public API."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class MedicalCondition(StrEnum):
    """Canonical Medical Abstracts target classes."""

    NEOPLASMS = "neoplasms"
    DIGESTIVE = "digestive system diseases"
    NERVOUS = "nervous system diseases"
    CARDIOVASCULAR = "cardiovascular diseases"
    GENERAL = "general pathological conditions"


class TriageRequest(BaseModel):
    """One medical abstract to classify for prioritization support."""

    text: str = Field(
        min_length=1,
        description="English medical abstract to classify.",
        examples=["The abstract describes a cardiovascular condition."],
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        """Reject whitespace-only values and trim request boundaries."""
        stripped = text.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped


class TriageResponse(BaseModel):
    """Primary label, all selected labels, independent scores and backend."""

    label: MedicalCondition
    labels: list[MedicalCondition]
    scores: dict[MedicalCondition, float]
    backend: str

    @model_validator(mode="after")
    def validate_scores(self) -> TriageResponse:
        """Require every canonical class and normalized probabilities."""
        if set(self.scores) != set(MedicalCondition):
            raise ValueError("scores must contain every canonical class")
        if any(score < 0 or score > 1 for score in self.scores.values()):
            raise ValueError("scores must be probabilities between 0 and 1")
        if self.label not in self.labels:
            raise ValueError("primary label must be included in labels")
        return self


class ExplainTerm(BaseModel):
    """One TF-IDF term's contribution to the predicted class score."""

    term: str
    tfidf: float
    weight: float
    contribution: float


class ExplainResponse(BaseModel):
    """Demo-only breakdown of one classification: preprocessing and terms."""

    label: MedicalCondition
    labels: list[MedicalCondition]
    scores: dict[MedicalCondition, float]
    backend: str
    preprocessed_text: str
    top_terms: list[ExplainTerm] | None = None


class SampleText(BaseModel):
    """One real example abstract used to seed the interactive demo."""

    label: MedicalCondition
    text: str


class HealthResponse(BaseModel):
    """Model loading status exposed by the health check."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    backend: str | None = None


class ServiceMetadata(BaseModel):
    """Stable service metadata returned by the root endpoint."""

    service: str
    version: str
