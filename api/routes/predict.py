"""Ruta publica de prediccion."""

from fastapi import APIRouter

def create_router(handler):
    router = APIRouter(tags=["prediction"])
    router.add_api_route(
        "/predict",
        handler,
        methods=["POST"],
        name="predict_risk",
    )
    return router
