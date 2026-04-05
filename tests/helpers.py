from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from src.tools.mock_server import OfficialRepoMockServer, build_default_session


@contextmanager
def running_mock_server(
    frame_count: int = 3,
    *,
    mode: str = "batch",
    **server_kwargs,
) -> Iterator[OfficialRepoMockServer]:
    server = OfficialRepoMockServer(
        session=build_default_session(frame_count=frame_count),
        mode=mode,
        **server_kwargs,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()
