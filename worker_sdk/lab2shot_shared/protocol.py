"""The protocol between the core and a worker, one definition for both sides: the job file's schema, the prefix of a
worker's event lines, what a message code looks like, and Failure (an error for the user as a code and its parameters,
for code both sides run). Paths in messages are written relative to the Lab2Shot folder (shown)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PREFIX = "@@lab2shot "


# 类型字母-模块-含义: the one rule for both sides (the core's catalogue imports it; the letters are lab2shot/messages LEVELS)
CODE = re.compile(r"^([EWNIBP])-([A-Z][A-Z0-9]*)-([A-Z][A-Z0-9]*)$")
# What a source scanner looks for (lab2shot check, tools/messages_web.py): a quoted string shaped like a code, any
# capital letter and any module. It only finds; CODE decides. A candidate CODE rejects is a malformed code in the
# source, which lab2shot check reports, so the finder is deliberately looser than the rule.
CODE_CANDIDATE = re.compile(r"""["']([A-Z]-[A-Z0-9]+-[A-Z0-9]+)["']""")


SCHEMA = "lab2shot.job/1"

# Where a job's files are: every path a worker is given is explicit and absolute, never the process's current folder
# (the process runs in the extension's original repository, which is never written to).
#   the job folder   the folder of the job file the core sends (the job command, or the one argument of a run by hand):
#                    an absolute path; frames and inputs in job.json are relative to it, and the worker writes only in it
#   raw results      RAW, inside the job folder
#   repo, weights    the extension's install location, the same for every job of a process: the environment variables
#                    REPO_ENV and WEIGHTS_ENV (engine/resident.py worker_environment sets them), absolute paths
RAW = "raw"
REPO_ENV = "LAB2SHOT_REPO_DIR"
WEIGHTS_ENV = "LAB2SHOT_WEIGHTS_DIR"
#   核预算         这次计算最多用几个线程：核心的「保留核心数」留出来给浏览器和别的程序之后剩下的
#                  （lab2shot/process.py worker_cpus；进程已经被绑在那几个核上了）。自己带线程参数的项目（COLMAP 的
#                  num_threads）要读它，不然库自己按「机器有几个核」开线程，绑核之后就是超订
CPU_BUDGET_ENV = "LAB2SHOT_CPU_BUDGET"


PROJECT_DIR = Path(__file__).resolve().parents[2]  # the Lab2Shot folder (the SDK is installed from it, editable)


class Failure(Exception):
    """fail() as an exception: a code (E-) and its parameters. For helpers the core imports too, where fail() would
    end the server (exr.write_exr, motion), and for errors a caller may still handle; a job that ends with one
    reports it as fail() does (serving._job). str(): the code and parameters (the words are the core's)."""

    def __init__(self, code: str, /, **params: Any):
        if not CODE.match(code):
            raise ValueError(f"not a message code: {code!r}")
        super().__init__(code)
        self.code, self.params = code, params

    def __str__(self) -> str:
        return f"{self.code} {json.dumps(self.params, ensure_ascii=False, default=str)}"


def shown(path: str | Path) -> str:
    """A path as messages the user sees write it: relative to the Lab2Shot folder, never absolute (a file outside
    the folder by its name)."""
    path = Path(path).absolute()
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.name


# --------------------------------------------------------------------------- the protocol

# What a worker process says on its standard output, one line each: PREFIX + JSON {"type": <event>, <fields>}. Anything
# else is log text (the job's worker.log). `status` stands for the loaded-models report
# ({"models": [{name, gpu_mb, ram_mb, on_gpu}], "vram_mb"}) spread into the event.
#   stage      name                   stage(): a named stage of the job starts
#   progress   done, total, message   progress(): how far the stage is
#   message    code, params           say(): a W-, N- or I- message for the user (the core writes the words from its catalogue)
#   done       result                 write_result(): raw/result.json is written, the job produced its files
#   failed     code, params           fail(), a Failure, the SDK (E-WORKER-GPUARCH, E-WORKER-OFFLINE): the job stops with
#                                     this E- message, the node shows it
#   nothing    code, params           nothing(): the job found nothing to give (N- message), the node gives empty outputs
#   job_end    ok, error, oom, incompatible_gpu, status
#                                     serve --serve, after each job: how one job of a kept process ended (error: its
#                                     code or the exception; oom: run once more alone)
#   offloaded  status                 serve --serve, the offload command: the kept models left the GPU
#
# What the core sends a kept process (worker.py --serve) on its standard input, one JSON line each {"cmd": ...}:
#   job        job, keep_free_gb      run the job file `job` (an absolute path); answered by job_end
#   offload    keep_free_gb           move the kept models off the GPU; answered by offloaded
#   exit                              end the process
#
# How a worker process exits: 0 the job ended (done, or nothing), or a kept process was told to exit; 1 the job
# failed (fail() ends with its code as the exit message), a kept process ends after a failed job, or the core it
# belongs to went away.
