"""RFC 7807 ``application/problem+json`` error handling.

Every error the API returns has the same shape, so a client never has to guess
whether a failure is `{"error": ...}`, `{"detail": ...}` or an HTML stack trace.
The domain exception hierarchy in :mod:`forecaster.exceptions` maps onto it
directly, which is why no router needs its own ``try/except``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from forecaster.exceptions import ForecasterError
from forecaster.logging import get_logger

log = get_logger(__name__)

PROBLEM_JSON = "application/problem+json"


def problem_response(
    status_code: int,
    code: str,
    detail: str,
    *,
    request: Request | None = None,
    **extra: Any,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"https://stockforecaster.dev/errors/{code}",
        "title": code.replace("_", " ").title(),
        "status": status_code,
        "detail": detail,
    }
    if request is not None:
        body["instance"] = str(request.url.path)
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            body["request_id"] = request_id
    if extra:
        body["details"] = extra
    return JSONResponse(status_code=status_code, content=body, media_type=PROBLEM_JSON)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ForecasterError)
    async def _domain_error(request: Request, exc: ForecasterError) -> JSONResponse:
        # 5xx is our fault and gets a stack trace; 4xx is the caller's and does not.
        if exc.http_status >= 500:
            log.error("domain_error", code=exc.code, detail=exc.message, exc_info=exc)
        else:
            log.info("client_error", code=exc.code, detail=exc.message)
        return problem_response(
            exc.http_status, exc.code, exc.message, request=request, **exc.details
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return problem_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_error",
            "Request validation failed.",
            request=request,
            errors=[
                {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
                for e in exc.errors()
            ],
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=request.url.path, exc_info=exc)
        # Never leak internals to the client; the request_id ties it to the log.
        return problem_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An unexpected error occurred.",
            request=request,
        )
