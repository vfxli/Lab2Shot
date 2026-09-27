"""The executor: runs an extension's install steps (plan.py) with one sink of events (events.py), the same for the
server's background task (server/installs.py: a farm task, cancelled with it) and the command line.

Each step records its state in the target environment's install_state: done with the fingerprint it was done for, or
failed with its message. Running again skips what is done and goes on from the failed step. Nothing is built over the
live environment: a changed spec builds beside it, and only `switch` (after the self-check passed and once no running
job uses this extension) makes it live; the one before stays for `rollback`.

Live is what only the server knows (running jobs, kept-loaded worker processes, free memory it can make); the command
line passes the defaults (nothing running that it can see).
"""

from __future__ import annotations

import fcntl
import json
import os
import pathlib
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lab2shot_worker.serving import available_gb

from .. import config
from ..errors import MessageError, Unavailable, message_of
from ..extensions import gpu_archs
from ..extensions.spec import Extension, ExtensionPaths, InstallError, Weight, clean_environ
from ..messages import Msg
from . import envbuild, plan, sources
from .events import Cancelled, Sink
from .sources import Gated, NetworkFailure, Policy

SELFCHECK_TIMEOUT_S = 600
SWITCH_POLL_S = 5.0  # how often a switch waiting for running jobs looks again
READ_S = 0.5  # how often a quiet command looks whether the install was cancelled
IDLE_S = 900  # a git transfer (fetch / submodule, with --progress) that prints nothing this long is stalled: stopped and retried.
# 15 minutes rather than 3: git runs with --progress, and a normal transfer is never silent that long. Applies only to
# git (run with idle=True): when uv pip's stdout is a pipe it prints nothing while downloading a multi-GB wheel or
# building a package from source; applying this timeout would kill it as stalled, restart from scratch and finally
# report a network failure. In those cases the install either waits or lets the compile error report itself


@dataclass
class Live:
    """What the running server knows. busy(name): jobs running now that use the extension; switched(name): its new
    environment went live (end kept-loaded processes of the old one); free_ram(gb): make memory by ending idle kept
    processes."""

    busy: Callable[[str], int] = field(default=lambda name: 0)
    switched: Callable[[str], None] = field(default=lambda name: None)
    free_ram: Callable[[float], None] = field(default=lambda gb: None)


@contextmanager
def locked(ext: Extension):
    """One install of an extension at a time on this machine (the server's queue and the command line alike)."""
    root = ext.paths.root
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".install.lock").open("w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise Unavailable(Msg("E-INSTALL-LOCKED", title=ext.title)) from exc
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class Context:
    """One install's working state: the extension, where it builds, its install record there, the events."""

    def __init__(self, ext: Extension, paths: ExtensionPaths, sink: Sink, policy: Policy, live: Live, force: bool):
        self.ext, self.paths, self.sink, self.policy, self.live, self.force = ext, paths, sink, policy, live, force
        self.state = plan.read_state(paths)
        current = plan.read_state(ext.paths)
        if paths != ext.paths:  # a new environment beside the live one: what is true of shared files carries over
            for key in ("weights", "verified", "post"):
                if key in current and key not in self.state:
                    self.state[key] = current[key]

    # ------------------------------------------------------------------ the record

    def save(self) -> None:
        verified = self.state.get("verified", {})  # the checksum cache speaks only for files still there
        for key in [k for k in verified if not (self.paths.root / k).exists()]:
            del verified[key]
        self.paths.root.mkdir(parents=True, exist_ok=True)
        tmp = self.paths.state_file.with_name(self.paths.state_file.name + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.paths.state_file)

    def mark_building(self) -> None:
        self.state["plan"] = plan.env_fingerprint(self.ext)
        self.save()
        ptr = plan.pointer(self.ext)
        ptr["building"] = plan.slot(self.paths)
        plan.write_pointer(self.ext, ptr)

    def record_env(self) -> None:
        freeze = self.run(["uv", "pip", "freeze", "--python", self.paths.python], quiet=True)
        self.state["env"] = {"python": self.ext.env.python, "fingerprint": plan.env_fingerprint(self.ext),
                             "packages": freeze.splitlines()}
        self.save()

    # ------------------------------------------------------------------ commands

    def run(self, args: list, cwd: Path | None = None, env: dict | None = None, network: bool = False,
            quiet: bool = False, idle: bool = False) -> str:
        """Run a command at the lowest priority, its output into the log (unless `quiet`); returns the output.
        NetworkFailure when it failed and (a `network` command) its last lines clearly say the network was why
        (sources.network_said: whole phrases, HTTP status phrases; a compile error is never one); InstallError
        otherwise; Cancelled when the install was cancelled meanwhile (the command is stopped). `idle`: a command
        that keeps printing while it transfers (git --progress) is stalled when it prints nothing for IDLE_S: stopped
        and raised as NetworkFailure so it is retried. The output is read on the calling thread, looking at the cancel
        every READ_S while the command is quiet: no thread of its own. The command runs in a session of its own, so
        stopping it stops everything it started (uv → setup.py → ninja → nvcc; git → git-remote-https) too."""
        args = [str(a) for a in args]
        self.sink.check()
        if not quiet:
            self.sink.log("$ " + " ".join(args))
        if env is None:
            # not os.environ: the core's VIRTUAL_ENV / PYTHONPATH and the pip / uv index settings of the server
            # user's shell must not leak into the extension's package installation and compilation
            # (spec.clean_environ, the same rule used by workers and the self-check)
            env = clean_environ()
        proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                preexec_fn=_lowest_priority, start_new_session=True)
        assert proc.stdout is not None
        out: list[str] = []
        tail: deque[str] = deque(maxlen=40)

        def keep(raw: bytes) -> None:
            """One line of output: kept for the caller, for the failure's tail, and logged."""
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            out.append(line)
            tail.append(line)
            if not quiet and line.strip():
                self.sink.log(line)

        pending = b""
        stalled = False  # the git transfer was silent for IDLE_S: on a degraded network it hangs, and the install would stay at this step forever unless it is killed
        ended = False  # EOF read: the command ended by itself. Leaving here before that (cancel, stall, Ctrl-C) must stop it together with its children
        last = time.monotonic()
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(proc.stdout, selectors.EVENT_READ)
                while True:
                    if self.sink.cancel.is_set():
                        break
                    if not ready.select(READ_S):
                        if idle and time.monotonic() - last > IDLE_S:
                            stalled = True
                            break
                        continue
                    last = time.monotonic()
                    chunk = os.read(proc.stdout.fileno(), 1 << 16)
                    if not chunk:
                        ended = True
                        break
                    *lines, pending = (pending + chunk).split(b"\n")
                    for raw in lines:
                        keep(raw)
            if pending and not self.sink.cancel.is_set():
                keep(pending)
        finally:
            if not ended:
                _end_group(proc)
            code = proc.wait()
            proc.stdout.close()
        self.sink.check()
        if stalled:
            self.sink.log(f"[{IDLE_S} 秒没有任何输出：当作网络卡住，停掉这一次]")
            raise NetworkFailure(f"no output for {IDLE_S} s")
        if code != 0:
            said = "\n".join(tail)
            if network and sources.network_said(said):
                raise NetworkFailure(said[-400:])
            raise InstallError(Msg("E-INSTALL-COMMAND", code=code, command=" ".join(args)[:300]))
        return "\n".join(out)

    def retry_command(self, args: list, what: str, env: dict | None = None, cwd: Path | None = None, idle: bool = False) -> str:
        """`run` with network=True, retried on a network failure (sources.retry). `idle`: see `run`; only for
        commands that print while they transfer (git); never for uv pip install, which is silent while it downloads
        a big wheel or builds from source."""
        return sources.retry(lambda: self.run(args, cwd=cwd, env=env, network=True, idle=idle), what, self.sink, self.policy)

    def wait_memory(self, need_gb: float) -> None:
        """Before a heavy step: the machine's reserve plus `need_gb` free (ending idle kept-loaded models first).
        Bounded: a machine that does not have that much at all (E-INSTALL-NOMEMORY), or does not free it within
        WAIT_MEMORY_S, fails the step instead of waiting for ever."""
        from lab2shot_shared.memory import total_gb

        want = float(config.settings().value("memory.keep_free_gb")) + need_gb
        if total_gb() < want:
            raise InstallError(Msg("E-INSTALL-NOMEMORY", want=want, total=total_gb()))
        said = False
        began = time.time()
        while available_gb() < want:
            if time.time() - began > WAIT_MEMORY_S:
                raise InstallError(Msg("E-INSTALL-NOMEMORY", want=want, total=available_gb()))
            self.live.free_ram(want)
            if available_gb() >= want:
                break
            if not said:
                self.sink.say(Msg("N-INSTALL-WAITMEMORY", free=available_gb(), want=want))
                said = True
            self.sink.cancel.wait(10)
            self.sink.check()


def _lowest_priority() -> None:
    try:
        os.nice(19)
    except OSError:
        pass


def _end_group(proc: subprocess.Popen) -> None:
    """Stop the command and everything it started: SIGTERM to its process group (it is its own session, so the group
    is exactly its tree), then SIGKILL to whatever is still there after 10 s. Terminating only the command itself is
    not enough: uv's setup.py → ninja → nvcc would go on compiling, git's git-remote-https would go on fetching, and the
    retry would run beside them."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            pass
        os.killpg(proc.pid, signal.SIGKILL)  # processes ignoring SIGTERM, and the children they left behind
    except ProcessLookupError:
        pass  # the whole group is already gone


# ------------------------------------------------------------------ steps


def step_repo(ctx: Context) -> None:
    ext, paths = ctx.ext, ctx.paths
    for src, dest in [(ext.source, paths.repo), *((s, paths.root / folder) for folder, s in ext.extra_sources.items())]:
        # modified after installation: not silently overwritten (the original repository must not be modified, and
        # silently overwriting someone's changes loses work); stop and list the changed files. Use --force to reset
        if (dirty := changed_files(dest)) and not ctx.force:
            raise InstallError(Msg("E-INSTALL-REPOCHANGED", repo=dest.name, count=len(dirty),
                              files="、".join(dirty[:5]) + ("……" if len(dirty) > 5 else "")))
        name = src.url.removesuffix('.git').rsplit('/', 1)[-1]
        if _head(dest) == src.commit:
            # already at the pinned commit: even with `--force` it is not deleted and re-fetched (running tasks import it,
            # and a failure midway could not be undone); only the changes are reverted, which achieves the same result
            # in one step and keeps the code readable at all times
            if dirty and ctx.force:
                ctx.run(["git", "-C", str(dest), "checkout", "--", "."], env=sources.git_env())
                ctx.run(["git", "-C", str(dest), "clean", "-fd"], env=sources.git_env())
                ctx.sink.log(f"↺ {name} @ {src.commit[:10]}：改过的 {len(dirty)} 个文件放回原样（--force）")
            else:
                ctx.sink.log(f"✓ {name} @ {src.commit[:10]}")
            continue
        if dest != paths.repo and dest.exists():
            # extra sources (dinov3, sam2, eigen, ...) have a single folder shared by the live environment and the one
            # being built, and sources.checkout starts with rmtree: checking out in place would delete the code running
            # tasks import, unrecoverable even by rollback. The main repository is checked out side by side as
            # repo-<commit> with the pointer swapped at switch; this is the minimal equivalent: the new commit is
            # checked out beside as <folder>.new, and step_switch swaps it in once tasks finish (the old copy is renamed
            # .prev and kept for rollback)
            staging = _extra_new(dest)
            if _head(staging) != src.commit:
                if _head(_extra_prev(dest)) == src.commit:  # the version used before a rollback: reuse it without downloading
                    if staging.exists():
                        shutil.rmtree(staging)
                    _extra_prev(dest).rename(staging)
                else:
                    sources.checkout(src, staging, ctx.sink, ctx.policy, ctx.run)
            ctx.sink.log(f"↻ {name} @ {src.commit[:10]}：已检出到 {staging.name}，启用那一步再换上去（正在用的那份不动）")
            continue
        sources.checkout(src, dest, ctx.sink, ctx.policy, ctx.run)
    missing = [p for p in ext.submodules if not (paths.repo / p).is_dir() or not any((paths.repo / p).iterdir())]
    if missing:
        ctx.retry_command(["git", "-C", paths.repo, "submodule", "update", "--init", "--recursive", "--depth", "1", "--progress", "--", *missing],
                          "子模块", env=sources.git_env(), idle=True)
    ctx.state["repo"] = {"url": ext.source.url, "commit": ext.source.commit, "dir": paths.repo_dir}
    ctx.save()


def _extra_new(dest: Path) -> Path:
    """Where a new commit of an extra source waits until `switch` (beside the live folder)."""
    return dest.with_name(dest.name + ".new")


def _extra_prev(dest: Path) -> Path:
    """The extra source's folder before the last switch, kept for rollback."""
    return dest.with_name(dest.name + ".prev")


def _pending_extras(ext: Extension) -> dict[str, Path]:
    """Extra sources checked out beside the live folder (step_repo) and not yet switched in: folder -> its .new."""
    root = ext.paths.root
    return {folder: _extra_new(root / folder) for folder, src in ext.extra_sources.items()
            if _head(_extra_new(root / folder)) == src.commit and _head(root / folder) != src.commit}


def _swap_extras(ext: Extension, extras: dict[str, Path]) -> list[str]:
    """Each waiting .new becomes the folder; the folder before becomes .prev (an older .prev goes). Returns the folders
    that have a .prev to roll back to. Two renames per folder, each atomic; jobs of this extension are done by now."""
    swapped = []
    for folder, new in extras.items():
        live = ext.paths.root / folder
        if live.exists():
            prev = _extra_prev(live)
            if prev.exists():
                shutil.rmtree(prev)
            live.rename(prev)
            swapped.append(folder)
        new.rename(live)
    return swapped


def _exchange(a: Path, b: Path) -> None:
    """Folders `a` and `b` change places (rollback of an extra source: the folder and its .prev)."""
    tmp = a.with_name(a.name + ".swap")
    a.rename(tmp)
    b.rename(a)
    tmp.rename(b)


def _head(dest: Path) -> str:
    if not (dest / ".git").exists():
        return ""
    return subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True,
                          env=sources.git_env()).stdout.strip()


# Artifacts left by running and compiling code do not count as modifications: bytecode caches, test caches, package
# metadata, and extension modules compiled into the repository by build scripts (compiled artifacts written back into a
# tracked checkout, such as SegAnyMo's sam2/_C.so). They are runtime products, not edits to the original code; counting
# them would make the next environment install report modified source code and push users toward --force
RUNTIME_LEFTOVERS = ("__pycache__/", ".pytest_cache/", ".ipynb_checkpoints/", ".egg-info/", ".pyc", ".pyo", ".so", ".pyd")


def changed_files(dest: Path) -> list[str]:
    """Files of the original repository modified after installation (the original repository must not be modified).

    A pinned commit only guarantees which version was fetched at install time; if someone edits it afterwards, HEAD is
    unchanged while the environment no longer matches the code, and the next reinstall would discard the edits. Files
    are counted as git reports them: modified, deleted and added. Runtime leftovers such as bytecode caches are not
    counted (RUNTIME_LEFTOVERS): they show the code was used, not edited. For a directory that is not a git repository
    (some extensions ship unpacked code) the result is empty, as there is no original state to compare with."""
    if not (dest / ".git").exists():
        return []
    # core.quotepath=false: otherwise git escapes Chinese and other non-ASCII file names as octal, making the user
    # message unreadable
    said = subprocess.run(["git", "-C", str(dest), "-c", "core.quotepath=false", "status", "--porcelain",
                           "--untracked-files=normal"], capture_output=True, text=True, env=sources.git_env()).stdout
    names = (line[3:].strip() for line in said.splitlines() if line.strip())
    return sorted(n for n in names if not any(part in n for part in RUNTIME_LEFTOVERS))


def step_weights(ctx: Context) -> None:
    from ..extensions import manual

    ext = ctx.ext
    rows = ctx.state.setdefault("weights", {})
    pending: list[Weight] = []
    for w in ext.weights:
        ctx.sink.check()
        if w.kind == "manual":  # downloaded by hand into the inbox (preflight made sure the required ones are there)
            state = manual.state(w.source)
            ctx.sink.say(Msg("I-INSTALL-MANUALOK", title=manual.item_of(ext, w.source).title) if state == "ready"
                         else manual.explain(w.source, state))
            continue
        dest = ext.paths.weights / w.dest
        pins = ctx.state.setdefault("weights_pinned", {})  # what each weight was fetched as: a changed pin fetches again
        if rows.get(w.key) == "ok" and dest.exists() and not ctx.force and pins.get(w.key) == _pin_of(w):
            if w.kind == "url" and w.sha256:
                sources.share(w.sha256, dest)  # an identical file another extension already verified: keep one copy
            if w.kind != "url" or not w.sha256 or _pinned(ctx, w, dest):
                ctx.sink.log(f"✓ {w.key}")
                continue
            # the file exists but is not the currently pinned version (the declared sha256 changed, the name did not):
            # it is not broken and is not deleted, since the live environment still relies on it. Continue with
            # _download -> sources.fetch_checked: the new file is downloaded beside it and swapped in once verified
            ctx.sink.log(f"↻ {w.key}：钉住的 sha256 变了，下载新的那一版")
        try:
            _download(ctx, w, dest)
            rows[w.key] = "ok"
            pins[w.key] = _pin_of(w)
        except Gated:
            rows[w.key] = "pending"
            if w.option is None:
                pending.append(w)
            ctx.sink.say(Msg("W-INSTALL-GATEDPENDING", weight=w.key, page=w.page))
        ctx.save()
    if pending:
        raise InstallError(Msg("E-INSTALL-GATED", weights=[w.key for w in pending], page=pending[0].page))


def _pin_of(w: Weight) -> str:
    """What pins a weight's contents: the snapshot's repository, revision and files; an archive's or file's address and
    sha256. Recorded with the download, so a changed declaration (a new revision, a new archive without a sha) is fetched
    again: the "ok" in `rows` alone would keep an hf / zip / sha-less url weight at its old version for ever."""
    return "|".join([w.kind, w.source, w.revision, w.sha256, ",".join(w.files)])


def _download(ctx: Context, w: Weight, dest: Path) -> None:
    if w.kind == "hf":  # a snapshot has no file hashes to check a mirror against: official only
        from huggingface_hub import get_token, snapshot_download
        from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

        # tokens are sent only to huggingface.co (the sources.request rule: never to a mirror). With HF_ENDPOINT pointing
        # at a mirror, huggingface_hub would send the machine's login token along, so gated repositories, or any download
        # on a machine with a login, use the official site; only token-less downloads follow HF_ENDPOINT
        endpoint = sources.HF if (w.gated or get_token()) else None

        def snapshot() -> None:
            try:
                _cancellable(ctx, lambda: snapshot_download(w.source, revision=w.revision, local_dir=dest,
                                                            allow_patterns=list(w.files) or None, endpoint=endpoint))
            except GatedRepoError as exc:
                raise Gated() from exc
            except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError) as exc:  # permanent: the declaration is wrong, no retry helps
                raise InstallError(Msg("E-INSTALL-DOWNLOAD", url=w.page, sources=["huggingface.co"], detail=str(exc)[:200])) from exc
            except HfHubHTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in (401, 403):
                    if w.gated:
                        raise Gated() from exc
                    raise InstallError(Msg("E-INSTALL-FORBIDDEN", url=w.page, code=status, page=w.page)) from exc  # an access request, not the network
                if status in (408, 425, 429, 500, 502, 503, 504):
                    raise ConnectionError(str(exc)) from exc
                raise InstallError(Msg("E-INSTALL-DOWNLOAD", url=w.page, sources=["huggingface.co"], detail=str(exc)[:200])) from exc

        sources.retry(snapshot, w.key, ctx.sink, ctx.policy)
    elif w.kind == "url":
        if not sources.share(w.sha256, dest):
            sources.fetch_checked(w.source, dest, w.sha256, ctx.sink, ctx.policy, gated=w.gated)
        _verify(ctx, w, dest)
    elif w.kind == "zip":
        import zipfile

        archive = ctx.ext.paths.weights / "_downloads" / Path(w.source).name
        if not sources.share(w.sha256, archive):
            sources.fetch_checked(w.source, archive, w.sha256, ctx.sink, ctx.policy, gated=w.gated)
        _verify(ctx, w, archive)
        from ..io.files import unpack

        unpack(archive, dest)  # the one unpacker (io/files.py: keeps nothing outside the archive, tar never escapes)
    else:
        raise InstallError(Msg("E-INSTALL-WEIGHTKIND", kind=w.kind))


def _cancellable(ctx: Context, fn: Callable[[], object]) -> None:
    """`fn()` (a library call with no cancel of its own: snapshot_download) on a thread of its own, the install's cancel
    looked at every READ_S meanwhile: Cancelled ends the step at once (the thread finishes its download on its own and
    is not waited for, as huggingface_hub cannot be stopped mid-file; the next attempt resumes what it wrote). Calling it
    on this thread would make a cancel wait until a snapshot of many GB has finished downloading."""
    failed: list[BaseException] = []

    def work() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 (carried to the calling thread, raised there)
            failed.append(exc)

    thread = threading.Thread(target=work, name="hf-snapshot", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(READ_S)
        ctx.sink.check()
    if failed:
        raise failed[0]


def _verify(ctx: Context, w: Weight, path: Path) -> None:
    """A file just downloaded (or linked) against its sha256 (when the weight gives one); a mismatch is a broken
    download: the file is deleted. A file already in place is judged with `_pinned` instead (a mismatch there is a
    different pinned version, kept until the new one arrives)."""
    if w.sha256 and not _pinned(ctx, w, path):
        path.unlink()
        ctx.state.setdefault("verified", {}).pop(path.relative_to(ctx.ext.paths.root).as_posix(), None)
        ctx.save()
        raise InstallError(Msg("E-INSTALL-CHECKSUM", weight=w.key, name=ctx.ext.name))


def _pinned(ctx: Context, w: Weight, path: Path) -> bool:
    """Whether the file at `path` is the one the weight pins (its sha256). Checked files are remembered by size and
    modification time, so installing again does not hash gigabytes again; a match is also recorded for sharing
    between extensions (sources.remember). Never deletes anything."""
    if not w.sha256:
        return True
    checked = ctx.state.setdefault("verified", {})
    key = path.relative_to(ctx.ext.paths.root).as_posix()
    st = path.stat()
    stamp = [st.st_size, st.st_mtime_ns, w.sha256]
    if checked.get(key) != stamp:
        if not sources.matches(path, w.sha256):
            return False
        checked[key] = stamp
        ctx.save()
    sources.remember(w.sha256, path)
    return True


def step_post(ctx: Context) -> None:
    ctx.ext.post_install(ctx.run, ctx.paths)
    ctx.state["post"] = True
    ctx.save()


def step_selfcheck(ctx: Context) -> None:
    """In the new environment, on the CPU, with no footage: the worker SDK, the pinned torch (and a tiny tensor sum)
    and the modules the extension declares import; its required weights are there; its GPU architectures recorded."""
    ext, paths = ctx.ext, ctx.paths
    env_fp = plan.env_fingerprint(ext)
    try:
        _selfcheck(ctx, env_fp)
    except MessageError as exc:
        ctx.state["selfcheck"] = {"ok": False, "fingerprint": env_fp, "time": time.time(), "message": exc.message.json()}
        ctx.save()
        raise


WAIT_MEMORY_S = 30 * 60  # a heavy step waits for memory this long at most


def _own_kernels(ext, paths) -> tuple[str, ...]:
    """Modules compiled by this extension, which the self-check also imports (derived from the scanned .so paths, not
    from declarations).

    `EnvSpec.imports` is optional and most extensions with compiled artifacts leave it empty, so it cannot be relied on
    alone. Typical failure: `sam2/_C.so` was compiled before an environment upgrade and no longer loads afterwards, and
    upstream swallows the exception with `except`; the self-check passes while the component is missing and masks are
    silently left unfilled. Derivation rules: see `extensions/gpu_archs.compiled_modules`.
    Anything that cannot be derived is skipped: under-reporting is preferred to blocking an install with a false name."""
    from ..extensions import gpu_archs

    try:
        # scan the environment being built (`paths`), not the live one (`ext.paths`): kernels pip compiled in the new
        # side-by-side environment are in `paths.venv`, and scanning the live one would include none of them
        rec = gpu_archs.probe(paths.python, paths.root, "")
    except Exception:  # noqa: BLE001 (a failed probe must not block the install; it is reported on its own path)
        return ()
    roots = [f"{paths.venv.name}/lib/python{ext.env.python}/site-packages"]
    if ext.import_repo is not None:
        roots.append(f"{paths.repo.name}/{ext.import_repo}".rstrip("/"))
    roots.append(paths.repo.name)
    for extra in (ext.run_env(paths, gpu="").get("PYTHONPATH", "") or "").split(":"):
        if extra.startswith(str(paths.root)):
            roots.append(str(pathlib.Path(extra).relative_to(paths.root)))
    # the packages that came as wheels (EnvSpec.torch: torch, torchvision, …) are not this extension's own build: the
    # self-check imports them whole above, and their compiled part is not always an importable module (torchvision's
    # `_C.so` is loaded by torchvision itself through torch.ops; `import torchvision._C` fails on every version since 0.17)
    wheels = {p.split("==")[0] for p in ext.env.torch}
    return tuple(m for m in gpu_archs.compiled_modules(rec.kernels, tuple(dict.fromkeys(roots)), paths.root)
                 if m.split(".")[0] not in wheels)


def _selfcheck(ctx: Context, env_fp: str) -> None:
    ext, paths = ctx.ext, ctx.paths
    if not paths.python.exists() or ctx.state.get("env", {}).get("fingerprint") != env_fp:
        raise InstallError(Msg("E-INSTALL-NOENV", title=ext.title))
    modules = ["lab2shot_worker", *(m for m in ("torch", "torchvision") if any(p.split("==")[0] == m for p in ext.env.torch)),
               *ext.env.imports, *_own_kernels(ext, paths)]
    # the very environment a worker of this extension then runs in (Extension.run_env, engine/resident.py's too),
    # with every card hidden: the check proves what will run, not a second environment made up here; in particular
    # this checkout's worker SDK, not whichever checkout pip installed into the environment
    env = ext.run_env(paths, gpu="")
    env["LAB2SHOT_SELFCHECK_MODULES"] = json.dumps(modules)
    # offline: an import that would reach Hugging Face for a config or a weight proves nothing about this environment and
    # hangs on a machine without the network; what the extension needs is on disk after step_weights, or the check fails
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    ctx.sink.log(f"$ {paths.python} -m lab2shot_worker.selfcheck  # {' '.join(modules)}")
    stdout, stderr = _selfcheck_process(ctx, [str(paths.python), "-m", "lab2shot_worker.selfcheck"], env, modules)
    line = next((l for l in reversed(stdout.splitlines()) if l.startswith("LAB2SHOT_SELFCHECK ")), "")
    if not line:
        raise InstallError(Msg("E-INSTALL-SELFCHECK", modules=modules, detail=(stderr or stdout).strip()[-300:]))
    found = json.loads(line.split(" ", 1)[1])
    if found["failed"]:
        name, why = next(iter(found["failed"].items()))
        raise InstallError(Msg("E-INSTALL-SELFCHECK", modules=list(found["failed"]), detail=f"{name}: {why}"))
    lacking = [w.key for w in ext.weights if w.kind != "manual" and w.option is None and ctx.state.get("weights", {}).get(w.key) != "ok"]
    if lacking:
        raise InstallError(Msg("E-INSTALL-SELFCHECKWEIGHTS", weights=lacking))
    archs = gpu_archs.probe(paths.python, paths.root, env_fp)
    ctx.state["gpu_archs"] = archs.to_json()
    ctx.state["selfcheck"] = {"ok": True, "fingerprint": env_fp, "time": time.time(), "python": found["python"],
                              "torch": found.get("torch", "")}
    ctx.save()
    ctx.sink.say(Msg("I-INSTALL-SELFCHECKOK", title=ext.title, modules=modules, archs=list(archs.archs or ())))


def _selfcheck_process(ctx: Context, args: list[str], env: dict, modules: list[str]) -> tuple[str, str]:
    """The self-check command: its stdout and stderr, waited for while looking at the install's cancel every READ_S
    (a plain subprocess.run(timeout=...) would make a cancel wait for the process to end or the timeout to run out; for
    a hanging import that is the whole SELFCHECK_TIMEOUT_S). On cancel or after SELFCHECK_TIMEOUT_S the whole process group
    is stopped (its own session, like `run`)."""
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out, \
            tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(args, env=env, cwd=ctx.paths.root, stdout=out, stderr=err, text=True,
                                preexec_fn=_lowest_priority, start_new_session=True)
        deadline = time.monotonic() + SELFCHECK_TIMEOUT_S
        try:
            while proc.poll() is None:
                if ctx.sink.cancel.is_set():
                    _end_group(proc)
                    ctx.sink.check()
                if time.monotonic() > deadline:
                    _end_group(proc)
                    raise InstallError(Msg("E-INSTALL-SELFCHECK", modules=modules, detail=f"{SELFCHECK_TIMEOUT_S} s"))
                try:
                    proc.wait(READ_S)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                _end_group(proc)
        out.seek(0)
        err.seek(0)
        return out.read(), err.read()


def step_switch(ctx: Context) -> None:
    """The new environment goes live once no running job uses the extension; the one before stays for rollback,
    older ones are removed. Extra sources checked out beside their live folder (step_repo) take its place here too."""
    ext, paths = ctx.ext, ctx.paths
    live = ext.paths
    ptr = plan.pointer(ext)
    extras = _pending_extras(ext)
    if paths == live and not extras:  # nothing new was built: it stays as it is
        if "building" in ptr:
            ptr.pop("building")
            plan.write_pointer(ext, ptr)
        return
    said = False
    while (n := ctx.live.busy(ext.name)) > 0:
        if not said:
            ctx.sink.say(Msg("N-INSTALL-WAITJOBS", title=ext.title, count=n))
            said = True
        ctx.sink.cancel.wait(SWITCH_POLL_S)
        ctx.sink.check()
    if paths != live:
        before = plan.slot(live) if live.python.exists() else None
        ptr = {"current": plan.slot(paths), "previous": before if before != plan.slot(paths) else ptr.get("previous"),
               "switched": time.time()}
    else:  # only extra sources change: the environment pointer stays as it is
        ptr = {k: v for k, v in ptr.items() if k not in ("building", "extras_previous")} | {"switched": time.time()}
    if extras:
        # which folders have a .prev to go back to: rollback() exchanges them again. A later switch that touches no
        # extra source drops the key: the environment before *that* switch ran with the folders as they are now
        ptr["extras_previous"] = _swap_extras(ext, extras)
    plan.write_pointer(ext, ptr)
    ctx.live.switched(ext.name)
    _tidy(ext, ptr)
    if paths != live:
        ctx.sink.say(Msg("I-INSTALL-SWITCHED", title=ext.title))
    if extras:
        ctx.sink.say(Msg("I-INSTALL-EXTRASWITCHED", title=ext.title, folders=list(extras)))


def _tidy(ext: Extension, ptr: dict) -> None:
    """Environments and checkouts neither live nor kept for rollback; a .new of an extra source that is already the
    live folder's commit (left over)."""
    keep_env = {e.get("env", "") for e in (ptr.get("current"), ptr.get("previous")) if e}
    keep_repo = {e.get("repo", "repo") for e in (ptr.get("current"), ptr.get("previous")) if e}
    root = ext.paths.root
    for d in root.glob(".venv*"):
        version = "" if d.name == ".venv" else d.name[len(".venv-"):]
        if version not in keep_env:
            shutil.rmtree(d, ignore_errors=True)
            ExtensionPaths(root, version).state_file.unlink(missing_ok=True)
    for d in root.glob("repo*"):
        if d.is_dir() and d.name not in keep_repo:
            shutil.rmtree(d, ignore_errors=True)
    for folder, src in ext.extra_sources.items():
        stale = _extra_new(root / folder)
        if stale.is_dir() and _head(root / folder) == src.commit:
            shutil.rmtree(stale, ignore_errors=True)


RUNNERS: dict[str, Callable[[Context], None]] = {
    "repo": step_repo, "env": envbuild.step_env, "packages": envbuild.step_packages, "build": envbuild.step_build,
    "weights": step_weights, "post": step_post, "selfcheck": step_selfcheck, "switch": step_switch,
}


def _done(step: plan.Step, ctx: Context) -> bool:
    if step.always:
        return False
    if step.id in plan.ENV_STEPS:
        # the environment finished for this spec (plan.built_for: the one judgement, the card's too) has every one of
        # its steps done; one still being built has the steps it recorded
        return plan.built_for(ctx.paths) == step.fingerprint or (ctx.paths.python.exists() and _step_done(ctx, step))
    if step.id == "repo":
        # an extra source is done when its folder is at the pinned commit, or its .new is (waiting for `switch`)
        return ctx.state.get("repo", {}).get("commit") == ctx.ext.source.commit and _head(ctx.paths.repo) == ctx.ext.source.commit \
            and all(_head(ctx.paths.root / f) == s.commit or _head(_extra_new(ctx.paths.root / f)) == s.commit
                    for f, s in ctx.ext.extra_sources.items())
    return _step_done(ctx, step)


def _step_done(ctx: Context, step: plan.Step) -> bool:
    rec = ctx.state.get("steps", {}).get(step.id, {})
    return rec.get("state") == "done" and rec.get("fingerprint") == step.fingerprint


def install(ext: Extension, sink: Sink, *, force: bool = False, live: Live | None = None, only: tuple[str, ...] = (),
            policy: Policy | None = None) -> ExtensionPaths:
    """Every step of `ext`'s install (or just `only`), from where it stopped; returns the environment it built."""
    live = live or Live()
    policy = policy or Policy.from_settings()
    with locked(ext):
        paths = ext.paths if only == ("selfcheck",) else plan.target(ext, force)
        ctx = Context(ext, paths, sink, policy, live, force)
        todo = [s for s in plan.steps(ext) if not only or s.id in only]
        sink.emit({"type": "plan", "steps": [{"id": s.id, "label": s.label} for s in todo]})
        for step in todo:
            sink.check()
            if not force and _done(step, ctx):
                sink.step(step.id, "skipped")
                continue
            sink.step(step.id, "running")
            try:
                RUNNERS[step.id](ctx)
            except Cancelled:
                sink.step(step.id, "cancelled")
                raise
            except Exception as exc:
                if isinstance(exc, MessageError):
                    message = message_of(exc)
                elif sources.transient(exc):  # its retries are used up: reported as a network failure, not 「意外的错误」
                    message = Msg("E-INSTALL-NETWORK", what=step.label, detail=str(exc)[:200] or type(exc).__name__)
                else:
                    message = Msg("E-INSTALL-PROBLEM", step=step.label, detail=str(exc)[:300] or type(exc).__name__)
                ctx.state.setdefault("steps", {})[step.id] = {"state": "failed", "fingerprint": step.fingerprint,
                                                              "time": time.time(), "message": message.json()}
                ctx.save()
                sink.step(step.id, "failed", message)
                raise StepFailed(step.id, message) from exc
            ctx.state.setdefault("steps", {})[step.id] = {"state": "done", "fingerprint": step.fingerprint, "time": time.time()}
            ctx.save()
            sink.step(step.id, "done")
        return paths


class StepFailed(MessageError):
    """A step failed: which, and its message (the install goes on from this step next time)."""

    def __init__(self, step: str, message: Msg):
        super().__init__(message)
        self.step = step


# ------------------------------------------------------------------ rollback and uninstall (administrator's actions)


def previous(ext: Extension) -> ExtensionPaths | None:
    """The environment kept for rollback, when it is still there."""
    prev = plan.paths_of(ext, plan.pointer(ext).get("previous"))
    return prev if prev is not None and prev.python.exists() else None


def rollback(ext: Extension, live: Live | None = None) -> None:
    """The environment before the last switch goes live again (and the one just rolled back from stays, so it can go
    forward again). Extra sources that switch exchanged (`extras_previous`) change places with their .prev again,
    both ways, so a second rollback goes forward again with them too."""
    live = live or Live()
    prev = previous(ext)
    root = ext.paths.root
    extras = [f for f in plan.pointer(ext).get("extras_previous", ()) if f in ext.extra_sources and _extra_prev(root / f).is_dir()]
    if prev is None and not extras:
        raise Unavailable(Msg("N-INSTALL-NOPREVIOUS", title=ext.title))
    if n := live.busy(ext.name):
        raise Unavailable(Msg("N-INSTALL-INUSE", title=ext.title, count=n))
    with locked(ext):
        ptr = plan.pointer(ext)
        for folder in extras:
            _exchange(root / folder, _extra_prev(root / folder))
        if prev is not None:
            current = ptr.get("current") or plan.slot(ext.paths)
            ptr = {"current": plan.slot(prev), "previous": current, "switched": time.time()}
        else:
            ptr = {k: v for k, v in ptr.items() if k != "extras_previous"} | {"switched": time.time()}
        if extras:
            ptr["extras_previous"] = extras
        plan.write_pointer(ext, ptr)
    live.switched(ext.name)


def uninstall(ext: Extension, live: Live | None = None) -> None:
    """Environments, checkouts, extra code, caches and install records go; model files stay (other extensions may
    share them by hash, and a reinstall does not download them again)."""
    live = live or Live()
    if n := live.busy(ext.name):
        raise Unavailable(Msg("N-INSTALL-INUSE", title=ext.title, count=n))
    root = ext.paths.root
    with locked(ext):
        for d in [*root.glob(".venv*"), *root.glob("repo*"), root / "cache",
                  *(root / (f + tail) for f in ext.extra_sources for tail in ("", ".new", ".prev", ".swap"))]:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        for f in [*root.glob("install_state*.json"), *root.glob("cuda_toolkit_overrides*.txt"), root / plan.ACTIVE_FILE]:
            f.unlink(missing_ok=True)
    live.switched(ext.name)
