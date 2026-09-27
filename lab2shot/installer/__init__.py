"""The extension installer: one module for every extension, driven by what each declares (adapters/<name>/extension.py:
source, env, weights, gated flags, manual files, imports for the self-check). No extension has install code of its
own; Extension.post_install is the one registered hook.

    plan.py       the steps an extension's declarations give, each with a fingerprint; where it builds; when an
                  environment is finished for a spec (built_for)
    preflight.py  the checklist before anything starts (hand downloads, licences, Hugging Face access, disk, GPUs)
    sources.py    official sources and checked mirrors, retries with backoff, resumable downloads
    envbuild.py   the environment, packages and build steps
    run.py        the executor: steps from where they stopped, self-check, switch, rollback, uninstall
    adopt.py      an environment built by hand: check it against the declaration, then record it (no building)
    events.py     what an install tells whoever watches it: the farm task it runs as (TaskSink), the console

It keeps no queue and starts no thread: the server runs an install as a background task of the farm
(server/installs.py INSTALL, farm/tasks.py: one at a time, cancellable, stopped when the farm closes); the command line
runs run.install itself.
"""

from .events import Cancelled, ConsoleSink, Recorder, Sink, TaskSink
from .plan import LABELS
from .run import Live, StepFailed, install, previous, rollback, uninstall

__all__ = ["LABELS", "Cancelled", "ConsoleSink", "Live", "Recorder", "Sink", "StepFailed", "TaskSink", "install",
           "previous", "rollback", "uninstall"]
