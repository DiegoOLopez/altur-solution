"""
Configuración global de la aplicación.

Utiliza ``pydantic-settings`` para cargar variables de entorno y
valores por defecto. Los valores se leen automáticamente de variables
de entorno con el mismo nombre (case-insensitive) o del archivo
``.env`` si existe.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación.

    Atributos:
        app_name:           Nombre de la aplicación (usado en logs).
        app_version:        Versión semántica del servicio.
        audio_storage_root: Ruta base para almacenar archivos de audio
                            temporales en disco (sólo en desarrollo).
    """

    app_name: str = "Altur Solution API"
    app_version: str = "0.1.0"
    audio_storage_root: str = "audio"


# Instancia global de configuración. Se importa en otros módulos como:
#     from backend.core.config import settings
settings = Settings()