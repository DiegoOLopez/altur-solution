from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación (cargada con pydantic-settings)."""

    app_name: str = "Altur Solution API"
    app_version: str = "0.1.0"
    audio_storage_root: str = "audio"


settings = Settings()