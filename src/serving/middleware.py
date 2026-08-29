import json
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response


CHINESE_SUPPRESSOR_QUALNAME = "src.models.vllm_logits_processors.ChineseTokenSuppressor"
GENERATION_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})


def approved_processor() -> list[dict[str, object]]:

    return [
        {
            "qualname": CHINESE_SUPPRESSOR_QUALNAME,
            "args": [],
            "kwargs": {},
        }
    ]


async def enforce_chinese_suppression(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:

    request_path = request.url.path
    root_path = request.scope.get("root_path", "")
    if root_path and request_path.startswith(root_path):
        path_without_root = request_path[len(root_path) :]
        if not path_without_root or path_without_root.startswith("/"):
            request_path = path_without_root or "/"

    if request.method != "POST" or request_path not in GENERATION_PATHS:
        return await call_next(request)

    try:
        payload = json.loads(await request.body())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return await call_next(request)

    if not isinstance(payload, dict):
        return await call_next(request)

    payload["logits_processors"] = approved_processor()
    request._body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request.scope["headers"] = [
        (name, value)
        for name, value in request.scope.get("headers", [])
        if name.lower() != b"content-length"
    ]
    return await call_next(request)
