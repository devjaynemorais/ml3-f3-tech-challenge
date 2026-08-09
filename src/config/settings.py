"""Application settings loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração de runtime da API — o que varia por ambiente de deploy.

    Hiperparâmetros de treino (features, modelo, split) ficam em
    `config/config.yaml` — não aqui — porque não são algo que se muda por
    ambiente de deploy, e sim por experimento.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Onde o serving procura os artefatos treinados (nomes de arquivo vêm do
    # config.yaml; o diretório-base é configurável por ambiente)
    model_artifacts_path: str = "models/artifacts"
    model_onnx_path: str = "models/onnx"

    # Backend de inferência da API: "sklearn" (pipeline joblib) ou "onnx"
    model_backend: str = "sklearn"

    # API serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
