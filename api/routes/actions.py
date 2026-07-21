"""Rutas para las acciones disparadas por n8n."""

from fastapi import APIRouter


def create_router(execute_handler):
    router = APIRouter(prefix="/actions", tags=["actions"])
    router.add_api_route(
        "/execute",
        execute_handler,
        methods=["POST"],
        name="execute_action",
    )
    return router
