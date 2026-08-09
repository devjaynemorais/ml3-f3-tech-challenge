"""Instrumentação Prometheus da API — contagem de requisições e latência.

Métricas expostas em `/metrics` (formato de texto do Prometheus):
  - `http_requests_total{method,path,status_code}` (Counter)
  - `http_request_duration_seconds{method,path}` (Histogram)
  - `triage_predictions_total{predicted_label}` (Counter) — usado pelo painel
    de distribuição de classificações no dashboard do Grafana.
"""

from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas pela API",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP em segundos",
    ["method", "path"],
)

PREDICTION_COUNT = Counter(
    "triage_predictions_total",
    "Total de classificações de urgência retornadas pela API",
    ["predicted_label"],
)


class PrometheusMiddleware:
    """Middleware ASGI que mede contagem e latência de toda requisição HTTP."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Encaminha a requisição e registra contagem/latência no Prometheus."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        # A API atual só tem rotas estáticas (sem parâmetros de path), então
        # a URL crua não gera explosão de cardinalidade na label `path`.
        path = request.url.path
        method = request.method
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            REQUEST_COUNT.labels(
                method=method, path=path, status_code=status_code
            ).inc()
            REQUEST_LATENCY.labels(method=method, path=path).observe(duration)


def metrics_response() -> Response:
    """Gera a resposta HTTP com as métricas no formato de texto do Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
