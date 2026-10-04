"""The framework's whole job flow through the fake host, in plain Python, against a running Lab2Shot server.

    python clients/common/tests/test_flow.py --server https://127.0.0.1:8766 --account account.json \
        --character Lab2Shot测试素材/AccuRIG-BLENDER-Caucasian_Realistic_man.fbx --project <folder>

(the server's own authority, when it serves HTTPS with one: --ca work/tls/ca.pem). No DCC module is imported anywhere
on the way: that is what this test proves. It also checks cancelling in the middle leaves nothing behind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # clients/common: lab2shot_dcc
sys.path.insert(0, HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--character", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--ca", default="")
    ap.add_argument("--tool", default="node~extract_skeleton")
    args = ap.parse_args()

    from lab2shot_dcc import paths

    if args.ca:  # what the download ships beside the package
        paths.bundled_ca = lambda: args.ca
    from fake_host import FakeHost
    from lab2shot_dcc.plugin import Plugin

    for name in list(sys.modules):
        assert not name.startswith(("maya", "hou", "nuke", "PySide", "shiboken")), name

    account = json.load(open(args.account, encoding="utf-8"))
    host = FakeHost(args.project)
    plugin = Plugin(host)
    said = {}

    def pump_until(cond, seconds):
        end = time.time() + seconds
        while not cond() and time.time() < end:
            host.pump()
        assert cond(), "timed out"

    plugin.login(args.server, account["username"], account["password"], lambda got, error: said.update(login=(got, error)))
    pump_until(lambda: "login" in said, 60)
    assert not said["login"][1], said["login"]
    print("login:", said["login"][0]["username"])

    plugin.refresh_tools(lambda error: said.update(tools=error))
    pump_until(lambda: "tools" in said, 120)
    assert not said["tools"], said["tools"]
    sources = sorted({t["source"] for t in plugin.tools})
    print("tools:", len(plugin.tools), sources, "formats:", [f["name"] for f in plugin.formats])

    character = host.add_object("Man (用户的角色)", "scene.character", os.path.abspath(args.character))
    host.selected = [character]
    order = plugin.ranked()
    first = {source: [t["id"] for t in tools[:3]] for source, _label, tools in order}
    print("ranked with a character selected:", first)

    tool = next(t for t in plugin.tools if t["id"] == args.tool)
    node = plugin.new_node()
    plugin.set_tool(node, tool)
    item = next(i for i in tool["inputs"] if not i.get("optional"))
    why = plugin.bind_selected(node, item)
    assert not why, why
    print("bound:", item["param"], "->", host.load(node)["bindings"][item["param"]]["label"])

    run = plugin.compute(node)
    pump_until(lambda: run.finished, 900)
    snap = run.snapshot()
    print("run:", snap["phase"], snap["text"], snap["error"])
    assert snap["phase"] == "done", snap
    state = host.load(node)
    version = state["versions"][-1]
    print("version:", version["version"], version["folder"], version["group_name"], version["namespace"])
    assert os.path.isdir(version["folder"]) and version["objects"]
    assert host.off_main_calls == 0, "a host call came from a background thread"
    imported = [host.scene[r] for r in version["objects"]]
    print("imported:", [(o["name"], o["kinds"]) for o in imported])

    # a second compute: a new version, a new group and namespace (numbered), nothing replaced
    run = plugin.compute(node)
    pump_until(lambda: run.finished, 900)
    state = host.load(node)
    assert [v["version"] for v in state["versions"]] == [1, 2], state["versions"]
    print("second version:", state["versions"][-1]["group_name"], state["versions"][-1]["namespace"])

    # the import is one undo step
    before = len(host.scene)
    host.undo()
    print("undo removed", before - len(host.scene), "objects")
    assert all(r not in host.scene for r in state["versions"][-1]["objects"])

    # fetching the last job again (B5)
    run = plugin.fetch(node)
    pump_until(lambda: run.finished, 300)
    assert run.snapshot()["phase"] == "done", run.snapshot()
    print("fetched again:", host.load(node)["versions"][-1]["version"])

    # the four sources, one contract: a saved graph (我的模板) and a local graph file run the same way
    lab = plugin.conn.session()
    graph = lab.tool(args.tool)["graph"]
    saved = lab._request("POST", "/api/my/templates", {"name": "插件测试 提取骨架", "graph": graph, "replace": True})
    local = os.path.join(args.project, "本地 节点图.json")
    with open(local, "w", encoding="utf-8") as f:
        json.dump({**graph, "meta": {"name": "本地的提取骨架"}}, f, ensure_ascii=False)
    said.pop("tools", None)
    plugin.add_local(local, lambda error: said.update(tools=error))
    pump_until(lambda: "tools" in said, 120)
    sources = sorted({t["source"] for t in plugin.tools})
    print("sources now:", sources)
    assert sources == ["local", "mine", "node", "preset"], sources
    fields = {s: sorted(next(t for t in plugin.tools if t["source"] == s)) for s in sources}
    print("same fields for every source:", all(set(f) >= {"id", "name", "source", "category", "inputs", "delivers", "exposed"}
                                               for f in fields.values()))
    for source in ("mine", "local"):
        tool = next(t for t in plugin.tools if t["source"] == source)
        other = plugin.new_node()
        plugin.set_tool(other, tool)
        host.selected = [character]
        assert not plugin.bind_selected(other, next(i for i in tool["inputs"] if not i.get("optional")))
        run = plugin.compute(other)
        pump_until(lambda: run.finished, 600)
        print(f"{source} tool:", run.snapshot()["phase"], host.load(other)["versions"][-1]["group_name"])
        assert run.snapshot()["phase"] == "done", run.snapshot()
    mine_id = next(t["id"] for t in plugin.tools if t["source"] == "mine")
    import urllib.parse

    lab._request("DELETE", "/api/my/templates/" + urllib.parse.quote(mine_id))
    del saved

    # cancel right after submitting: nothing half written stays
    base = os.path.dirname(version["folder"])
    before = sorted(os.listdir(base))
    run = plugin.compute(node)
    pump_until(lambda: run.snapshot()["phase"] in ("queued", "running", "uploading") or run.finished, 120)
    run.cancel()
    pump_until(lambda: run.finished, 120)
    print("cancelled:", run.snapshot()["phase"], run.snapshot()["text"])
    assert run.snapshot()["phase"] == "cancelled", run.snapshot()
    assert sorted(os.listdir(base)) == before, (before, os.listdir(base))
    print("FLOW OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
