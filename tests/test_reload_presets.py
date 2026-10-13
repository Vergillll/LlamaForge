"""reload_presets(): one models.ini re-read on either router route, against a
stub router on a real socket."""
import conftest_paths  # noqa: F401
import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import routes


class StubRouter(BaseHTTPRequestHandler):
    """A router over one preset. `disk` is models.ini, `seen` what the router
    last read from it; GET /models reports `seen` as the model's args.
    query: GET /models?reload=1 re-reads, POST /models/reload is `missing`.
    post:  POST /models/reload re-reads, GET /models ignores its query."""
    flavour, missing = "query", 404
    disk, seen = {}, {}
    reloads, requests = 0, []

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _reload(self):
        cls = type(self)
        cls.seen, cls.reloads = dict(cls.disk), cls.reloads + 1

    def do_GET(self):
        self.requests.append(("GET", self.path))
        if self.path.split("?")[0] != "/models":
            return self._send(404, {"error": "not found"})
        if self.flavour == "query" and self.path.endswith("?reload=1"):
            self._reload()
        self._send(200, {"data": [{"id": "m", "status": {"value": "unloaded",
                                                         "args": dict(self.seen)}}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self.requests.append(("POST", self.path))
        if self.path == "/models/reload" and self.flavour == "post":
            self._reload()
            return self._send(200, {"success": True})
        self._send(self.missing, {"error": "not found"})


class Base(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), StubRouter)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        p = mock.patch.object(routes, "cfg", return_value={
            "router_port": self.srv.server_address[1], "router_local_key": "k"})
        p.start()
        self.addCleanup(p.stop)

    def router_as(self, flavour, missing=404):
        StubRouter.flavour, StubRouter.missing = flavour, missing
        StubRouter.seen = {"n-cpu-moe": "14"}            # what the router started with
        StubRouter.disk = {"n-cpu-moe": "15"}            # a Save has since written the file
        StubRouter.reloads, StubRouter.requests = 0, []

    def args_seen(self):
        status, out = routes.router("/models")
        self.assertEqual(status, 200)
        return out["data"][0]["status"]["args"]


class ReloadPresetsTest(Base):
    def test_a_query_route_router_reloads_once(self):
        for missing in (404, 405):
            with self.subTest(missing=missing):
                self.router_as("query", missing)
                status, _ = routes.reload_presets()
                self.assertEqual(status, 200)
                self.assertEqual(StubRouter.reloads, 1)
                self.assertEqual(StubRouter.requests,
                                 [("POST", "/models/reload"), ("GET", "/models?reload=1")])
                self.assertEqual(self.args_seen(), {"n-cpu-moe": "15"})

    def test_a_post_route_router_reloads_once(self):
        self.router_as("post")
        status, out = routes.reload_presets()
        self.assertEqual((status, out), (200, {"success": True}))
        self.assertEqual(StubRouter.reloads, 1)
        self.assertEqual(StubRouter.requests, [("POST", "/models/reload")])
        self.assertEqual(self.args_seen(), {"n-cpu-moe": "15"})

    def test_a_save_reloads_once_on_either_router(self):
        for flavour in ("query", "post"):
            with self.subTest(flavour=flavour), \
                 mock.patch.object(routes, "_model_status", return_value={"value": "unloaded"}):
                self.router_as(flavour)
                self.assertFalse(routes._reload_preset("m"))
                self.assertEqual(StubRouter.reloads, 1)
                self.assertEqual(self.args_seen(), {"n-cpu-moe": "15"})


class RouterDownTest(unittest.TestCase):
    def test_says_so_without_raising_or_retrying(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()                                         # nothing listens there now
        with mock.patch.object(routes, "cfg", return_value={"router_port": port}), \
             mock.patch.object(routes, "router", wraps=routes.router) as r:
            status, out = routes.reload_presets()
        self.assertEqual(status, 599)
        self.assertIn("could not reach the router to reload models.ini", out["error"])
        self.assertEqual(r.call_count, 1)


class CallSitesTest(unittest.TestCase):
    def test_every_reload_goes_through_the_helper(self):
        hits = []
        for root, _, files in os.walk(os.path.dirname(routes.__file__)):
            for name in sorted(files):
                if name.endswith(".py"):
                    with open(os.path.join(root, name), encoding="utf-8") as f:
                        hits += [name for line in f if '"/models?reload=1"' in line]
        self.assertEqual(hits, ["routes.py"])            # the fallback inside reload_presets


if __name__ == "__main__":
    unittest.main()
