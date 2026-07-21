"""Rutas que consume el panel web del tutor.

Se agrupan bajo /panel y comparten la dependencia de autenticacion, a
diferencia de las rutas que consume n8n, que permanecen abiertas para no
alterar el flujo de automatizacion.
"""

from fastapi import APIRouter, Depends


def create_router(
    listar_estudiantes,
    resumen_panel,
    explicar_prediccion,
    actualizar_caso,
    dependencia_auth,
):
    router = APIRouter(
        prefix="/panel",
        tags=["panel"],
        dependencies=[Depends(dependencia_auth)],
    )
    router.add_api_route(
        "/estudiantes",
        listar_estudiantes,
        methods=["GET"],
        name="listar_estudiantes",
    )
    router.add_api_route(
        "/resumen",
        resumen_panel,
        methods=["GET"],
        name="resumen_panel",
    )
    router.add_api_route(
        "/predicciones/{prediction_id}/explicacion",
        explicar_prediccion,
        methods=["GET"],
        name="explicar_prediccion",
    )
    router.add_api_route(
        "/casos/{prediction_id}",
        actualizar_caso,
        methods=["PATCH"],
        name="actualizar_caso",
    )
    return router
