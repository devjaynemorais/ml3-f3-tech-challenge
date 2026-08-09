from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.serving.metrics import REQUEST_COUNT, PrometheusMiddleware


async def _ping(request):
    return PlainTextResponse("pong")


def _build_app() -> Starlette:
    app = Starlette(routes=[Route("/ping", _ping)])
    app.add_middleware(PrometheusMiddleware)
    return app


def test_middleware_incrementa_contador_por_metodo_path_e_status() -> None:
    client = TestClient(_build_app())
    counter = REQUEST_COUNT.labels(method="GET", path="/ping", status_code=200)
    before = counter._value.get()

    client.get("/ping")

    after = counter._value.get()
    assert after == before + 1
