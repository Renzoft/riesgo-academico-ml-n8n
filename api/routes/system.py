"""Rutas operativas, historicas y de resultados."""

from fastapi import APIRouter


def create_router(root, health, model_info, history, register_outcome):
    router = APIRouter(tags=["system"])
    router.add_api_route("/", root, methods=["GET"], name="read_root")
    router.add_api_route(
        "/health", health, methods=["GET"], name="health"
    )
    router.add_api_route(
        "/model-info", model_info, methods=["GET"], name="model_info"
    )
    router.add_api_route(
        "/history", history, methods=["GET"], name="history"
    )
    router.add_api_route(
        "/outcomes",
        register_outcome,
        methods=["POST"],
        name="register_outcome",
    )
    return router
