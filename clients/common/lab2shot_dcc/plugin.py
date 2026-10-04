"""The plugin as the panel and the menu use it, one per DCC session: the connection, the tool list, and what each
Lab2Shot node does (pick a tool, bind inputs, set values, compute, cancel, fetch). Everything here is DCC-free: it
talks to the host only through lab2shot_dcc.host.Host.

Threads: methods that talk to the server (`refresh_tools`, `login`) run on a background thread and call back with the
answer through the host's run_on_main; the rest are called on the main thread and return at once.

The node's state (stored in the scene by the host, saved with it):
  tool         {id, source, name, file (a local graph)}
  values       what the user set, by outside name
  scene_values what the scene filled (「来自场景」), by outside name
  bindings     input outside name -> Binding (host.Binding)
  job          the last job: {id, server, state, …}
  versions     every result fetched: {version, job, folder, group, namespace, objects, imported, made, tool}
"""

from __future__ import annotations

import os
import threading
import traceback
import webbrowser

from . import catalog, connection, contract, jobs, log, safety, view_model
from .context import scene_context
from .guard import guarded
from .paths import text


class Plugin:
    def __init__(self, host):
        self.host = host
        log.setup(host.name)
        self.lang = connection.apply_language(host)  # the plugin's words and the server's, in the user's language
        self.conn = connection.Connection(app=host.name)
        self.tools: list[dict] = []
        self.formats: list[dict] = []
        self.categories: list[dict] = []
        self.colorspaces: list[str] = []
        self._language_listeners: list = []
        self.tools_error = ""
        self.loading = False
        self.runs: dict[str, jobs.Run] = {}
        host.on_exit(self.shutdown)

    def shutdown(self) -> None:
        """The DCC is quitting: every job here lets go (they go on on the server)."""
        for run in list(self.runs.values()):
            run.abandon()

    # ---- connection and tools (background threads, answers on the main thread)

    def _background(self, name: str, work, done) -> None:
        def run():
            try:
                got, error = work(), ""
            except Exception as exc:  # noqa: BLE001
                log.get().error("%s failed: %s\n%s", name, exc, traceback.format_exc())
                got, error = None, str(exc) or type(exc).__name__
            self.host.run_on_main(lambda: guarded(done, what=name)(got, error))

        threading.Thread(target=run, name=f"lab2shot-{name}", daemon=True).start()

    def set_language(self, choice: str) -> None:
        """The user's language setting (zh, en, or connection.LANG_AUTO: the DCC's), kept and used from now on: every
        listener (the panel, the DCC's menu) is told at once, and the tool list is asked again (the server's words —
        tool names, intros, categories — come in the language asked for)."""
        connection.write_settings(lang=choice)
        self.lang = connection.apply_language(self.host)
        for node, run in list(self.runs.items()):  # a finished run's last sentence was said in the old language: its
            if run.finished:  # result is on the node already, the sentence goes
                self.runs.pop(node, None)
        for listener in list(self._language_listeners):
            guarded(listener, what="language listener")()
        if self.conn.has_token():
            self.refresh_tools(lambda _error: [guarded(f, what="language listener")() for f in list(self._language_listeners)])

    def on_language(self, listener) -> None:
        """`listener()` (main thread) when the language changes, and again once the tool list came back in it."""
        if listener not in self._language_listeners:
            self._language_listeners.append(listener)

    def off_language(self, listener) -> None:
        if listener in self._language_listeners:
            self._language_listeners.remove(listener)

    def login(self, address: str, username: str, password: str, done) -> None:
        self._background("login", lambda: self.conn.login(address, username, password), done)

    def refresh_tools(self, done=None) -> None:
        """Fetch the tool list (and describe the local graph files); `done(error)` on the main thread."""
        if self.loading:
            return
        self.loading = True

        def work():
            if not self.conn.server:
                raise RuntimeError(text("dcc.plugin.not_connected"))
            why = connection.reachable(self.conn.server)
            if why:
                raise RuntimeError(why)
            lab = self.conn.session()
            got = catalog.fetch(lab)
            for path in catalog.local_files():
                try:
                    got["tools"].append(catalog.describe_local(lab, path))
                except Exception as exc:  # noqa: BLE001 - one unreadable file is said, the rest are listed
                    log.get().warning("local node graph %s does not read: %s", path, exc)
            return got

        def finish(got, error):
            self.loading = False
            if got is not None:
                self.tools, self.formats, self.colorspaces = got["tools"], got["formats"], got["colorspaces"]
                self.categories = got.get("categories") or []
            self.tools_error = error
            if done is not None:
                done(error)

        self._background("tools", work, finish)

    def add_local(self, path: str, done=None) -> None:
        catalog.remember_local(path)
        self.refresh_tools(done)

    def tool(self, state: dict) -> dict | None:
        """The tool entry of a node's state, from the list as fetched (None while not fetched or gone)."""
        tid = (state.get("tool") or {}).get("id")
        return next((t for t in self.tools if t.get("id") == tid), None) if tid else None

    def ranked(self) -> list:
        return catalog.ranked(self.tools, self.host.selection_types(), self.host.prefers, self.host.exports)

    # ---- 「最近用过」: per server (another server's tool ids are other tools), in the plugin's settings

    def recent(self) -> list[str]:
        got = connection.read_settings().get("recent_tools") or {}
        mine = got.get(self.conn.server) if isinstance(got, dict) else None
        return [str(t) for t in mine if isinstance(t, str)] if isinstance(mine, list) else []

    def remember_recent(self, tool_id: str) -> None:
        if not tool_id or not self.conn.server:
            return
        got = connection.read_settings().get("recent_tools")
        got = dict(got) if isinstance(got, dict) else {}
        got[self.conn.server] = view_model.recent_after(self.recent(), tool_id)
        connection.write_settings(recent_tools=got)

    # ---- nodes (main thread)

    def new_node(self) -> str:
        n = 1  # lab2shot1, lab2shot2 …: the first name nothing in the scene has
        while self.host.name_taken(f"lab2shot{n}"):
            n += 1
        node = self.host.create_node(f"lab2shot{n}")
        self.host.store(node, {"server": self.conn.server, "values": {}, "scene_values": {}, "bindings": {},
                               "versions": [], "job": {}})
        return node

    def _change(self, node: str, fn) -> dict:
        state = self.host.load(node)
        fn(state)
        self.host.store(node, state)
        return state

    def set_tool(self, node: str, tool: dict) -> dict:
        def change(state):
            state["tool"] = {"id": tool["id"], "source": tool.get("source"), "name": tool.get("name"),
                             "file": tool.get("file", "")}
            names = {x["name"] for x in tool.get("exposed") or []}
            state["values"] = {k: v for k, v in (state.get("values") or {}).items() if k in names}
            keys = contract.input_keys(tool)
            state["bindings"] = {k: v for k, v in (state.get("bindings") or {}).items() if k in keys}
        state = self._change(node, change)
        self.fill_from_scene(node)
        return state

    def bind_selected(self, node: str, item: dict) -> str:
        """Bind what is selected to one input; returns "" or why not."""
        binding = self.host.bind(item.get("type", ""), item.get("kinds") or [])
        if binding is None:
            return self.host.why_not(item.get("type", "")) or text("dcc.plugin.wrong_selection")
        binding = dict(binding)
        binding.setdefault("type", item.get("type", ""))
        # what goes up: the file the host exports the object as (Host.exports), else the object's own file
        sent = catalog.suffix(binding["type"], self.host.exports) or \
            os.path.splitext(str(binding.get("file") or ""))[1].lower()
        if not catalog.reads(item, sent):
            return text("dcc.plugin.wrong_format", host=self.host.label, suffix=sent,
                        accept=" ".join(item.get("accept") or []))

        def change(state):
            state.setdefault("bindings", {})[item["param"]] = binding
            if item.get("when"):
                state.setdefault("values", {}).update(item["when"])
        self._change(node, change)
        self.fill_from_scene(node)
        return ""

    def bind_file(self, node: str, item: dict, path: str | list[str]) -> None:
        """`path`: one file, or the frames picked together (jobs.picked_files: what is sent and what is shown)."""
        def change(state):
            files = sorted(path) if isinstance(path, list) else []
            state.setdefault("bindings", {})[item["param"]] = {
                "file": files[0] if files else path, **({"files": files} if files else {}),
                "label": jobs.files_label(files or path, item.get("widget") == "sequence"), "type": item.get("type", "")}
            if item.get("when"):
                state.setdefault("values", {}).update(item["when"])
        self._change(node, change)

    def unbind(self, node: str, param: str) -> None:
        self._change(node, lambda s: (s.get("bindings") or {}).pop(param, None))

    def set_value(self, node: str, name: str, value) -> None:
        def change(state):
            state.setdefault("values", {})[name] = value
        self._change(node, change)

    def reset_value(self, node: str, name: str) -> None:
        self._change(node, lambda s: (s.get("values") or {}).pop(name, None))

    def fill_from_scene(self, node: str) -> dict:
        """The scene's values of the tool's fixed names, 「来自场景」 (the user's own values stay above them)."""
        state = self.host.load(node)
        tool = self.tool(state)
        if tool is None:
            return state
        context = scene_context(self.host, state.get("bindings") or {})
        state["scene_values"] = contract.scene_fill(tool, context, self.colorspaces)
        self.host.store(node, state)
        return state

    # ---- jobs

    def busy(self, node: str) -> bool:
        run = self.runs.get(node)
        return run is not None and not run.finished

    def why_not_saved(self) -> str:
        """"" when the scene is saved; else why nothing can be computed or fetched yet (results always go
        next to the saved scene, never anywhere else)."""
        return text("dcc.plugin.save_first") if self.host.unsaved() else ""

    def _need_saved(self) -> None:
        why = self.why_not_saved()
        if why:
            raise RuntimeError(why)

    def compute(self, node: str, ask=None) -> jobs.Run:
        if self.busy(node):
            return self.runs[node]
        self._need_saved()
        state = self.fill_from_scene(node)
        tool = self.tool(state)
        if tool is None:
            raise RuntimeError(text("dcc.plugin.no_tool"))
        missing = [i.get("label") or i["param"] for i in tool.get("inputs") or []
                   if not i.get("optional") and i["param"] not in (state.get("bindings") or {})]
        if missing:
            raise RuntimeError(text("dcc.plugin.missing_inputs", inputs=text("list.sep").join(missing)))
        run = jobs.Run(self.host, self.conn, node, tool, ask=ask, colorspaces=self.colorspaces)
        self.runs[node] = run
        run.start()
        try:
            self.remember_recent(tool.get("id", ""))
        except OSError:  # the settings file not writable: 「最近用过」 is a convenience
            log.get().warning("recent tools not kept: %s", traceback.format_exc(limit=1))
        return run

    def import_version(self, node: str, version: int) -> jobs.Run:
        """Bring in a version that was fetched but not imported (its files are on this machine: nothing is downloaded
        again), as the version it is — the same import step a fetch runs (jobs.Run, `version`)."""
        if self.busy(node):
            return self.runs[node]
        self._need_saved()
        state = self.host.load(node)
        entry = next((v for v in state.get("versions") or [] if int(v.get("version") or 0) == int(version)), None)
        if entry is None:
            raise RuntimeError(text("dcc.plugin.no_version", version=f"v{int(version):03d}"))
        if entry.get("imported", True):
            raise RuntimeError(text("dcc.plugin.imported_already", version=f"v{int(version):03d}"))
        run = jobs.Run(self.host, self.conn, node, self.tool(state), version=int(version))
        self.runs[node] = run
        run.start()
        return run

    def fetch(self, node: str, ask=None) -> jobs.Run:
        """Fetch the node's last job's results again (after the scene was closed, or a dropped line)."""
        if self.busy(node):
            return self.runs[node]
        self._need_saved()
        state = self.host.load(node)
        job = (state.get("job") or {}).get("id") or next((v.get("job") for v in reversed(state.get("versions") or [])
                                                          if v.get("job")), "")
        if not job:
            raise RuntimeError(text("dcc.plugin.never_submitted"))
        run = jobs.Run(self.host, self.conn, node, self.tool(state), job=job, ask=ask)
        self.runs[node] = run
        run.start()
        return run

    def cancel(self, node: str) -> None:
        run = self.runs.get(node)
        if run is not None:
            run.cancel()

    # ---- 「在网页里打开」 (B4): the node's last job in the page, its newest result fetched when the page is done

    def web_address(self, state: dict | None) -> str:
        """The page address inside the server: the node's last job focused on the node the tool's signature names
        (`focus`: worked out by the server from the graph, the same for every client), or the editor when the node
        has no job yet."""
        job = ((state or {}).get("job") or {}).get("id")
        if not job:
            return "/"
        tool = self.tool(state) or {}
        focus = tool.get("focus")
        return f"/#job={job}" + (f"&focus={focus}" if focus else "")

    def open_in_web(self, node: str | None = None, say=None) -> None:
        """The embedded window (lab2shot_dcc.ui.web): a one-time address asked on a background thread, opened at once
        on the main thread; when the page is done, the newest finished job that follows this node's is fetched as a
        new version (fetch_follows)."""
        state = self.host.load(node) if node else {}
        to = self.web_address(state)
        lab = self.conn.session()

        def opened(url, error):
            if error:
                raise RuntimeError(error)
            from .ui import web

            def browser():
                self._background("web", lambda: lab.embed(to), lambda u, e: webbrowser.open(u) if u else None)

            web.open_job(self.host, url, lambda: self.fetch_follows(node, say) if node and "job=" in to else None,
                         browser, say)

        self._background("web", lambda: lab.embed(to), opened)

    def fetch_follows(self, node: str, say=None) -> None:
        """The newest finished job of this account that follows the node's job (directly, or along a chain of jobs
        each following the one before), fetched and imported as a new version like any other."""
        self._need_saved()
        state = self.host.load(node)
        mine = {(state.get("job") or {}).get("id")} | {v.get("job") for v in state.get("versions") or []}
        mine.discard(None)

        def find():
            history = self.conn.session().queue().get("history") or []
            follows = {h.get("id"): h.get("follows") or "" for h in history}

            def leads_home(job_id):
                seen = set()
                while job_id and job_id not in seen:
                    seen.add(job_id)
                    if follows.get(job_id) in mine:
                        return True
                    job_id = follows.get(job_id)
                return False

            done = [h for h in history if h.get("state") == "done" and h.get("id") not in mine and leads_home(h.get("id"))]
            done.sort(key=lambda h: -float(h.get("finished") or h.get("submitted") or 0))
            return done[0]["id"] if done else ""

        def found(job, error):
            if error:
                raise RuntimeError(error)
            if not job:
                if say:
                    say(text("dcc.plugin.nothing_new"))
                return
            run = jobs.Run(self.host, self.conn, node, self.tool(self.host.load(node)), job=job,
                           colorspaces=self.colorspaces)
            self.runs[node] = run
            run.start()

        self._background("follows", find, found)
