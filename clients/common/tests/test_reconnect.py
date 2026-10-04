"""The server going away while a job is submitted (a restart, a dropped line): the plugin says it is reconnecting,
tries a bounded number of times, and either goes on once the server is back or stops with a clear sentence — it never
waits for ever. A TCP proxy in front of the server stands for it being there or not.

    python clients/common/tests/test_reconnect.py --server-port 8766 --account account.json --character x.fbx \
        --project <folder> --ca work/tls/ca.pem
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


class Proxy:
    def __init__(self, upstream: int):
        self.upstream, self.socks, self.listener = upstream, [], None
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()

    def start(self):
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", self.port))
        self.listener.listen(32)
        threading.Thread(target=self._accept, args=(self.listener,), daemon=True).start()

    def stop(self):
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        for s in self.socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
                s.close()
            except OSError:
                pass
        self.socks = []

    def _accept(self, listener):
        while True:
            try:
                client, _ = listener.accept()
            except OSError:
                return
            try:
                server = socket.create_connection(("127.0.0.1", self.upstream))
            except OSError:
                client.close()
                continue
            self.socks += [client, server]
            for a, b in ((client, server), (server, client)):
                threading.Thread(target=self._pump, args=(a, b), daemon=True).start()

    @staticmethod
    def _pump(a, b):
        try:
            while True:
                data = a.recv(1 << 16)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.close()
                except OSError:
                    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-port", type=int, default=8766)
    ap.add_argument("--account", required=True)
    ap.add_argument("--character", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--ca", default="")
    args = ap.parse_args()
    from lab2shot_dcc import paths

    if args.ca:
        paths.bundled_ca = lambda: args.ca
    from fake_host import FakeHost
    from lab2shot_dcc.plugin import Plugin

    proxy = Proxy(args.server_port)
    proxy.start()
    account = json.load(open(args.account, encoding="utf-8"))
    host = FakeHost(args.project)
    plugin = Plugin(host)
    said = {}

    def pump_until(cond, seconds):
        end = time.time() + seconds
        while not cond() and time.time() < end:
            host.pump()
        return cond()

    plugin.login(f"https://127.0.0.1:{proxy.port}", account["username"], account["password"], lambda u, e: said.update(login=e))
    pump_until(lambda: "login" in said, 60)
    assert not said["login"], said["login"]
    plugin.refresh_tools(lambda e: said.update(tools=e))
    pump_until(lambda: "tools" in said, 300)
    tool = next(t for t in plugin.tools if t["id"] == "node~extract_skeleton")
    character = host.add_object("Man", "scene.character", os.path.abspath(args.character))
    host.selected = [character]
    node = plugin.new_node()
    plugin.set_tool(node, tool)
    assert not plugin.bind_selected(node, tool["inputs"][0])
    ok = True

    def watch(run, seconds):
        texts = []
        end = time.time() + seconds
        while not run.finished and time.time() < end:
            host.pump()
            t = run.snapshot()["text"]
            if not texts or texts[-1] != t:
                texts.append(t)
        return texts

    # A: the server away when 计算 is pressed, back 7 s later: reconnects and finishes
    proxy.stop()
    t0 = time.time()
    run = plugin.compute(node)
    threading.Timer(7, proxy.start).start()
    texts = watch(run, 300)
    snap = run.snapshot()
    good = snap["phase"] == "done" and any("重连" in t for t in texts)
    ok &= good
    print(("PASS" if good else "FAIL"), "A server back after 7 s:", snap["phase"], f"{time.time() - t0:.0f}s", texts[:4])

    # B: the server gone for good: stops within the bound, saying what to do
    proxy.stop()
    t0 = time.time()
    run = plugin.compute(node)
    texts = watch(run, 240)
    snap = run.snapshot()
    took = time.time() - t0
    good = run.finished and snap["phase"] == "failed" and "连不上" in snap["error"] and took < 120
    ok &= good
    print(("PASS" if good else "FAIL"), f"B server gone: stopped after {took:.0f}s:", snap["error"][:120], "| tries said:",
          [t for t in texts if "重连" in t][-1:])

    # C: the line drops while the job is followed (right after it was submitted), back 10 s later
    proxy.start()
    run = plugin.compute(node)
    pump_until(lambda: run.snapshot()["phase"] in ("queued", "running") or run.finished, 120)
    proxy.stop()
    threading.Timer(10, proxy.start).start()
    texts = watch(run, 300)
    snap = run.snapshot()
    good = snap["phase"] == "done"
    ok &= good
    print(("PASS" if good else "FAIL"), "C line dropped while following:", snap["phase"], [t for t in texts if "重连" in t][:2])
    proxy.stop()
    print("RECONNECT OK" if ok else "RECONNECT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
