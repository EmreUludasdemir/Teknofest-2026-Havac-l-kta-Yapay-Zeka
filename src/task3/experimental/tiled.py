from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(slots=True)
class ExperimentalTiledInference:
    enabled: bool = False

    def status(self) -> dict[str, object]:
        if not self.enabled:
            return {"active": False, "reason": "disabled"}
        if importlib.util.find_spec("sahi") is None:
            return {"active": False, "reason": "missing_dependency:sahi"}
        return {"active": True, "reason": "available"}
