"""The core's unit tests (standard-library unittest: no server, no GPU, no network).

Run them all from the repository root:

    uv run python -m unittest discover -s tests -t .      # or: uv run lab2shot check tests (with the page's tests)
    uv run python -m unittest tests.test_core             # one file

Every test runs against a work folder of its own (a temporary one, set here before anything reads the settings), never
the checkout's work/: a test that opens the database, writes the cache or a log touches nothing a server uses."""

import atexit
import os
import shutil
import tempfile

_WORK = tempfile.mkdtemp(prefix="lab2shot-tests-")
os.environ["LAB2SHOT_WORK_DIR"] = _WORK
atexit.register(shutil.rmtree, _WORK, True)
