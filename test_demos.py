"""Run real HAPEE processes and local mocks; no third-party packages required."""
import contextlib
import csv
import http.client
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
HAPROXY = os.environ.get("HAPROXY_BIN", "/opt/hapee-3.2/sbin/hapee-lb")


def request(port, payload=None, headers=None, path="/v1/chat/completions"):
    body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
    try:
        conn.request("POST" if payload is not None else "GET", path, body,
                     {"Content-Type": "application/json", **(headers or {})})
        response = conn.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        conn.close()


def completion(model="llama-3.1-8b"):
    return {"model": model, "messages": [{"role": "user", "content": "Hello"}]}


def backend_stats(port, backend):
    """HAProxy's own `show stat` rows for one backend, keyed by server name."""
    with socket.create_connection(("127.0.0.1", port), timeout=4) as sock:
        sock.sendall(b"show stat\n")
        raw = b"".join(iter(lambda: sock.recv(65536), b""))
    rows = csv.DictReader(io.StringIO(raw.decode().lstrip("# ")))
    return {row["svname"]: row for row in rows if row["pxname"] == backend}


class Processes:
    """Own only our children; retain their logs until fixture cleanup."""
    def __init__(self):
        self.stack = contextlib.ExitStack()

    def close(self):
        self.stack.close()

    def start(self, command, port):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port unavailable: {port}") from exc
        log = self.stack.enter_context(tempfile.TemporaryFile(mode="w+"))
        proc = subprocess.Popen(command, stdout=log, stderr=log)
        self.stack.callback(self.stop, proc)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.seek(0)
                raise RuntimeError(f"Process failed: {command}\n{log.read()}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.1):
                    # A pre-existing listener must not mask our child's bind failure.
                    time.sleep(.05)
                    if proc.poll() is None:
                        return proc
            except OSError:
                pass
            time.sleep(.02)
        raise RuntimeError(f"Process did not listen: {command}")

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


def start_demo(owner, name, servers, frontend):
    processes = Processes()
    owner.addClassCleanup(processes.close)
    directory = processes.stack.enter_context(tempfile.TemporaryDirectory())
    # Reserve all selected ports together; release each only when its child starts.
    reservations = {}
    for original in [frontend, "stats", *(port for port, _, _ in servers)]:
        sock = processes.stack.enter_context(socket.socket())
        sock.bind(("127.0.0.1", 0))
        reservations[original] = sock
    ports = {old: sock.getsockname()[1] for old, sock in reservations.items()}
    source = ROOT / name / "haproxy.cfg"
    if not source.is_file():
        raise AssertionError(f"Missing demo configuration: {source}")
    # Test-only stats socket: read server state from HAProxy instead of inferring it.
    config = source.read_text().replace(
        "global\n", "global\n    stats socket ipv4@127.0.0.1:stats\n", 1)
    if "stats socket" not in config:
        raise AssertionError(f"No global section to extend: {source}")
    for old, new in ports.items():
        config = config.replace(f"127.0.0.1:{old}", f"127.0.0.1:{new}")
    config_path = Path(directory) / "haproxy.cfg"
    config_path.write_text(config)
    subprocess.run([HAPROXY, "-c", "-f", str(config_path)], check=True,
                   capture_output=True, text=True)
    children = {}
    for port, model, replica in servers:
        reservations[port].close()
        children[replica] = processes.start(
            [sys.executable, str(ROOT / "mock_inference.py"), "--port", str(ports[port]),
             "--model", model, "--replica", replica], ports[port])
    reservations[frontend].close()
    reservations["stats"].close()
    processes.start([HAPROXY, "-db", "-f", str(config_path)], ports[frontend])
    return ports[frontend], children, ports


class ModelRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port, _, cls.ports = start_demo(cls, "model-routing", [
            (18101, "llama-3.1-8b", "llama-1"),
            (18102, "qwen2.5-7b", "qwen-1")], 18080)

    def test_model_routes_and_openai_shape(self):
        for model, replica in [("llama-3.1-8b", "llama-1"), ("qwen2.5-7b", "qwen-1")]:
            with self.subTest(model=model):
                status, headers, raw = request(self.port, completion(model))
                self.assertEqual(status, 200, raw)
                self.assertEqual(headers["x-mock-replica"], replica)
                body = json.loads(raw)
                self.assertEqual(body["object"], "chat.completion")
                self.assertEqual(body["model"], model)
                self.assertTrue(body["id"].startswith("chatcmpl-"))
                self.assertIsInstance(body["created"], int)
                self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
                self.assertEqual(body["choices"][0]["finish_reason"], "stop")
                self.assertIn("MOCK", body["choices"][0]["message"]["content"])

    def test_content_type_gate(self):
        for content_type in ["application/json; charset=utf-8", "application/json ; charset=utf-8",
                             "application/json\t;charset=utf-8", "Application/JSON"]:
            with self.subTest(content_type=content_type):
                status, _, raw = request(self.port, completion(), {"Content-Type": content_type})
                self.assertEqual(status, 200, raw)
        for content_type in ["application/jsonp", "application/json-patch+json",
                             "application/json charset=utf-8", "xapplication/json", "text/plain"]:
            with self.subTest(content_type=content_type):
                status, _, raw = request(self.port, completion(), {"Content-Type": content_type})
                self.assertEqual(status, 415)
                # The mock answers 415 too, with a JSON error; only the gateway's is empty.
                self.assertEqual(raw, b"")

    def test_unusable_models_are_rejected(self):
        for payload in [{}, {"model": "unknown"}, {"model": None}, {"model": 8},
                        {"model": ["llama-3.1-8b"]}, b'{"model":', b'not json']:
            with self.subTest(payload=payload):
                status, headers, _ = request(self.port, payload)
                self.assertEqual(status, 400)
                self.assertNotIn("x-mock-replica", headers)

    def test_mock_validates_openai_subset(self):
        for payload in [dict(completion(), stream=True), {"model": "llama-3.1-8b"},
                        dict(completion(), messages=[]),
                        dict(completion(), messages=[{"role": "user", "content": 42}]),
                        b'{"model":"llama-3.1-8b","model":"qwen2.5-7b","messages":[]}']:
            with self.subTest(payload=payload):
                self.assertEqual(request(self.port, payload)[0], 400)
        status, _, raw = request(self.ports[18101], completion("qwen2.5-7b"))
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(raw))

    def test_inspection_limit(self):
        payload = completion()
        payload["messages"][0]["content"] = "x" * 9000
        status, headers, _ = request(self.port, payload)
        self.assertEqual(status, 413)
        self.assertNotIn("x-mock-replica", headers)

    def test_body_limit_boundary_and_full_buffer(self):
        prefix = b'{"model":"llama-3.1-8b","messages":[{"role":"user","content":"'
        suffix = b'"}]}'
        raw = prefix + b'x' * (8192 - len(prefix) - len(suffix)) + suffix
        self.assertEqual(request(self.port, raw)[0], 200)
        self.assertEqual(request(self.port, raw + b' ')[0], 413)
        # The body fits the policy limit but not the buffer remaining after headers.
        status, headers, _ = request(self.port, raw, {"X-Padding": "x" * 25000})
        self.assertEqual(status, 413)
        self.assertNotIn("x-mock-replica", headers)

    def test_chunked_body_is_explicitly_unsupported(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=4)
        try:
            conn.request("POST", "/v1/chat/completions", iter([json.dumps(completion()).encode()]),
                         {"Content-Type": "application/json"}, encode_chunked=True)
            self.assertEqual(conn.getresponse().status, 411)
        finally:
            conn.close()

    def test_incomplete_body_times_out(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=4) as sock:
            sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n'
                         b'Content-Type: application/json\r\nContent-Length: 200\r\n\r\n'
                         b'{"model":"llama-3.1-8b"}')
            self.assertIn(b" 408 ", sock.recv(4096).split(b"\r\n")[0])

    def test_wrong_path_and_method(self):
        self.assertEqual(request(self.port, completion(), path="/other")[0], 404)
        self.assertEqual(request(self.port, path="/v1/chat/completions")[0], 405)


class ConsistentHashing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port, cls.children, cls.ports = start_demo(cls, "consistent-hashing", [
            (18201, "llama-3.1-8b", "gpu-1"),
            (18202, "llama-3.1-8b", "gpu-2"),
            (18203, "llama-3.1-8b", "gpu-3")], 18081)

    def test_affinity_and_replica_failure(self):
        def selected(key):
            status, headers, raw = request(
                self.port, completion(), {"X-Conversation-ID": key})
            self.assertEqual(status, 200, raw)
            return headers["x-mock-replica"]

        before = {f"conversation-{i}": selected(f"conversation-{i}") for i in range(120)}
        self.assertEqual(set(before.values()), {"gpu-1", "gpu-2", "gpu-3"})
        self.assertEqual({key: selected(key) for key in before}, before)
        canary = next(key for key, replica in before.items() if replica == "gpu-2")

        def stats():
            return backend_stats(self.ports["stats"], "gpu_replicas")

        def redispatches():
            return int(stats()["BACKEND"]["wredis"])

        # gpu-2 must still be in the ring when it stops, or the canary proves nothing.
        self.assertTrue(stats()["gpu-2"]["status"].startswith("UP"))
        self.assertEqual(redispatches(), 0)
        Processes.stop(self.children["gpu-2"])
        # The first request to find gpu-2 gone is redispatched once, with no same-server
        # retry, and its failed connect alone (not a health check) marks gpu-2 DOWN.
        owner = selected(canary)
        now = stats()
        self.assertEqual(int(now["BACKEND"]["wredis"]), 1, "The canary missed the failure window")
        self.assertEqual(int(now["BACKEND"]["wretr"]), 0)
        self.assertEqual(now["gpu-2"]["status"], "DOWN")
        self.assertEqual(int(now["gpu-2"]["hanafail"]), 1)
        after = {key: selected(key) for key in before}
        # gpu-2 really left the ring: nothing after the canary needed a redispatch.
        self.assertEqual(redispatches(), 1)
        # One move, not two: the redispatched canary already reached its new ring owner.
        self.assertEqual(after[canary], owner)
        self.assertNotIn("gpu-2", after.values())
        for key, replica in before.items():
            if replica != "gpu-2":
                self.assertEqual(after[key], replica, key)
        moved = sum(before[key] != after[key] for key in before)
        self.assertEqual(moved, list(before.values()).count("gpu-2"))
        print(f"\nFailover: {moved}/{len(before)} conversations moved; surviving assignments stable.",
              flush=True)

        # Without traffic, health checks alone must find a dead replica.
        Processes.stop(self.children["gpu-3"])
        deadline = time.monotonic() + 5
        while stats()["gpu-3"]["status"] != "DOWN":
            if time.monotonic() > deadline:
                self.fail("Health checks did not remove gpu-3 within 5 seconds")
            time.sleep(.1)
        self.assertEqual(int(stats()["gpu-3"]["hanafail"]), 0)
        self.assertEqual({selected(key) for key in before}, {"gpu-1"})
        self.assertEqual(redispatches(), 1)

    def test_missing_or_blank_affinity_is_rejected(self):
        for headers in [{}, {"X-Conversation-ID": ""}, {"X-Conversation-ID": "   "}]:
            with self.subTest(headers=headers):
                status, response_headers, _ = request(self.port, completion(), headers)
                self.assertEqual(status, 400)
                self.assertNotIn("x-mock-replica", response_headers)


class ProcessSafety(unittest.TestCase):
    def test_startup_failure_cleans_up_owned_children(self):
        children = Processes()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        proc = None
        try:
            proc = children.start([sys.executable, str(ROOT / "mock_inference.py"),
                                   "--port", str(port), "--model", "llama-3.1-8b",
                                   "--replica", "cleanup"], port)
            self.assertEqual(request(port, path="/health")[0], 200)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                other_port = reservation.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "Process failed"):
                children.start([sys.executable, "-c", "raise SystemExit(7)"], other_port)
        finally:
            children.close()
        self.assertIsNotNone(proc)
        self.assertIsNotNone(proc.poll())

    def test_occupied_port_fails_without_stopping_existing_listener(self):
        with socket.socket() as listener, contextlib.closing(Processes()) as children:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "Port unavailable"):
                children.start([sys.executable, str(ROOT / "mock_inference.py"),
                                "--port", str(port), "--model", "llama-3.1-8b",
                                "--replica", "collision"], port)
            self.assertEqual(listener.getsockname()[1], port)


if __name__ == "__main__":
    unittest.main()
