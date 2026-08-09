"""Modelos Pydantic de request/response da API de triagem."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Corpo da requisição de classificação de um laudo médico."""

    text: str = Field(
        ...,
        min_length=1,
        description="Texto do laudo/relato clínico a ser classificado.",
        examples=[
            "Paciente relata dor torácica intensa e súbita, "
            "irradiando para o braço esquerdo."
        ],
    )


class TriageResponse(BaseModel):
    """Resposta da classificação de urgência."""

    label: str = Field(description="Classe prevista: normal, atencao ou urgente.")
    scores: dict[str, float] = Field(description="Probabilidade por classe.")
    backend: str = Field(description="Backend de inferência usado: sklearn ou onnx.")
