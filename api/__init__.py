"""Capa publica de la API FastAPI.

La aplicacion se resuelve de forma diferida para que los esquemas puedan ser
usados por ML sin cargar TensorFlow ni crear dependencias circulares.
"""

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from api.main import app

        return app
    raise AttributeError(name)
