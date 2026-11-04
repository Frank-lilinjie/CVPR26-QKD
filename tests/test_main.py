import io
import json
import unittest
from contextlib import redirect_stdout

from qkd_mindspore.main import DEFAULT_CONFIG, main


class MainTests(unittest.TestCase):
    def test_dry_run_does_not_import_trainer(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--config", str(DEFAULT_CONFIG), "--dry-run"])
        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["framework"], "mindspore")
        self.assertEqual(payload["device_target"], "ascend")
        self.assertEqual(payload["ms_mode"], "pynative")
        self.assertEqual(payload["max_tasks"], 2)


if __name__ == "__main__":
    unittest.main()
