"""Ruta de monitoreo del comportamiento productivo."""

from fastapi import APIRouter


def create_router(handler):
    router = APIRouter(tags=["monitoring"])
    router.add_api_route(
        "/monitoring",
        handler,
        methods=["GET"],
        name="monitoring_report",
    )
    return router
