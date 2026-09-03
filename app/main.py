"""Entrypoint que conserva las rutas existentes y omite campos opcionales nulos."""

from starlette.routing import request_response

from .main_base import *  # noqa: F401,F403
from .main_base import app


for route in app.routes:
    if getattr(route, "path", None) in {"/predecir_codigos", "/predecir_auditoria"}:
        route.response_model_exclude_none = True
        route.app = request_response(route.get_route_handler())
