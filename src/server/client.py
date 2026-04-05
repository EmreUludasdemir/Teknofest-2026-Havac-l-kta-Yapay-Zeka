from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


@dataclass(slots=True)
class HttpResponse:
    status_code: int
    body: bytes
    headers: dict[str, str]

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.text)


class SimpleHttpClient:
    """Stdlib tabanli kucuk HTTP istemcisi."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
        timeout: float = 5.0,
    ) -> HttpResponse:
        req = request.Request(url=url, data=data, headers=headers or {}, method=method)
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return HttpResponse(
                    status_code=response.getcode(),
                    body=response.read(),
                    headers=dict(response.info()),
                )
        except error.HTTPError as exc:
            return HttpResponse(
                status_code=exc.code,
                body=exc.read(),
                headers=dict(exc.headers.items()),
            )

    def get_json(self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 5.0) -> HttpResponse:
        return self.request("GET", url, headers=headers, timeout=timeout)

    def get_bytes(self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 10.0) -> HttpResponse:
        return self.request("GET", url, headers=headers, timeout=timeout)

    def post_form(
        self,
        url: str,
        form_data: dict[str, str],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> HttpResponse:
        encoded = parse.urlencode(form_data).encode("utf-8")
        merged_headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if headers:
            merged_headers.update(headers)
        return self.request("POST", url, headers=merged_headers, data=encoded, timeout=timeout)

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> HttpResponse:
        encoded = json.dumps(payload).encode("utf-8")
        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)
        return self.request("POST", url, headers=merged_headers, data=encoded, timeout=timeout)
