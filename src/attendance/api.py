"""Thin ASGI layer (Starlette). All logic lives in AttendanceService, so swapping the web framework
(e.g. to FastAPI, which is built on Starlette) only touches this file."""
from __future__ import annotations

import json
import re

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import Settings, load_settings
from .service import AttendanceService, ServiceError
from .store import Store

DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _role(request: Request, *allowed: str) -> str:
    key = request.headers.get("x-api-key")
    if not key:
        raise ServiceError(401, "missing_api_key", "X-API-Key header is required")
    role = request.app.state.settings.api_keys.get(key)
    if role is None:
        raise ServiceError(401, "invalid_api_key", "API key not recognised")
    if role not in allowed:
        raise ServiceError(403, "forbidden", f"this endpoint needs role: {' or '.join(allowed)}")
    return role


async def _payload(request: Request) -> tuple[bytes | None, list | None, dict]:
    """Either multipart (image file + fields) or JSON (embedding + fields)."""
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        image = await upload.read() if upload is not None and hasattr(upload, "read") else None
        return image, None, {k: v for k, v in form.items() if k != "image"}
    if ctype.startswith("application/json"):
        try:
            data = json.loads(await request.body() or b"{}")
        except ValueError:
            raise ServiceError(400, "malformed_json", "body is not valid JSON") from None
        if not isinstance(data, dict):
            raise ServiceError(422, "validation_error", "body must be a JSON object")
        return None, data.pop("embedding", None), data
    raise ServiceError(415, "unsupported_media_type", "use multipart/form-data (image) or application/json")


def _svc(request: Request) -> AttendanceService:
    return request.app.state.service


async def health(request: Request):
    svc = _svc(request)
    return JSONResponse({"status": "ok", "people": len(svc.store.people()),
                         "models_loaded": svc._engine is not None})


async def people(request: Request):
    role = _role(request, "admin")
    if request.method == "GET":
        return JSONResponse({"people": _svc(request).store.people()})
    _, _, data = await _payload(request)
    return JSONResponse(_svc(request).create_person(str(data.get("external_id", "")).strip(),
                                                    str(data.get("name", "")).strip(), role), status_code=201)


async def person_templates(request: Request):
    role = _role(request, "admin")
    image, embedding, data = await _payload(request)
    force = str(data.get("force", "")).lower() in ("1", "true", "yes")
    out = _svc(request).enroll(request.path_params["person_id"], role, image=image, embedding=embedding, force=force)
    return JSONResponse(out, status_code=201)


async def delete_person(request: Request):
    role = _role(request, "admin")
    return JSONResponse(_svc(request).delete_person(request.path_params["person_id"], role))


async def recognize(request: Request):
    _role(request, "device", "admin")
    image, embedding, data = await _payload(request)
    device = str(data.get("device_id") or "unknown-device")
    return JSONResponse(_svc(request).recognize(device, image=image, embedding=embedding, ts=data.get("ts")))


async def report(request: Request):
    _role(request, "admin")
    day = request.path_params["day"]
    if not DATE_RX.match(day):
        raise ServiceError(422, "validation_error", "date must be YYYY-MM-DD")
    return JSONResponse(_svc(request).report(day))


async def purge(request: Request):
    role = _role(request, "admin")
    return JSONResponse(_svc(request).purge(role))


async def audit(request: Request):
    _role(request, "admin")
    return JSONResponse({"audit": _svc(request).store.q("SELECT * FROM audit ORDER BY id DESC LIMIT 200")})


def create_app(settings: Settings | None = None, store: Store | None = None, engine=None) -> Starlette:
    settings = settings or load_settings()

    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message, **exc.detail}},
                            status_code=exc.status)

    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"error": {"code": "http_error", "message": str(exc.detail)}}, status_code=exc.status_code)

    app = Starlette(routes=[
        Route("/health", health),
        Route("/v1/people", people, methods=["GET", "POST"]),
        Route("/v1/people/{person_id:int}/templates", person_templates, methods=["POST"]),
        Route("/v1/people/{person_id:int}", delete_person, methods=["DELETE"]),
        Route("/v1/recognize", recognize, methods=["POST"]),
        Route("/v1/attendance/{day}", report),
        Route("/v1/admin/purge", purge, methods=["POST"]),
        Route("/v1/admin/audit", audit),
    ], exception_handlers={ServiceError: service_error, HTTPException: http_error})
    app.state.settings = settings
    app.state.service = AttendanceService(settings, store or Store(settings.database), engine)
    return app
