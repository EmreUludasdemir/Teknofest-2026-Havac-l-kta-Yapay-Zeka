from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.run_phase10_operator_drill import read_appended_runtime_events


class Phase10OperatorDrillTests(unittest.TestCase):
    def test_read_appended_runtime_events_returns_only_new_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.jsonl"
            path.write_text(
                json.dumps({"event": "old"}) + "\n" + json.dumps({"event": "new"}) + "\n",
                encoding="utf-8",
            )
            offset = len((json.dumps({"event": "old"}) + "\n").encode("utf-8"))
            events = read_appended_runtime_events(path, offset)
            self.assertEqual(events, [{"event": "new"}])


if __name__ == "__main__":
    unittest.main()
