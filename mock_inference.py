"""Loopback-only OpenAI-style chat mock. Does not perform inference."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import uuid

MAX_BODY = 8192


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are unsupported")
        result[key] = value
    return result


class InferenceHandler(BaseHTTPRequestHandler):
    def send_json(self, status, value, replica=False):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if replica:
            self.send_header("X-Mock-Replica", self.server.replica)
        self.end_headers()
        self.wfile.write(raw)

    def error(self, status, message):
        self.send_json(status, {"error": {"message": message,
                       "type": "invalid_request_error", "param": None, "code": None}})

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        else:
            self.error(404, "Unknown endpoint")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.error(404, "Unknown endpoint")
            return
        if self.headers.get("Transfer-Encoding") or not self.headers.get("Content-Length"):
            self.error(411, "Content-Length required; chunked uploads unsupported")
            return
        if self.headers.get_content_type() != "application/json":
            self.error(415, "Content-Type must be application/json")
            return
        try:
            length = int(self.headers["Content-Length"])
        except ValueError:
            self.error(400, "Invalid Content-Length")
            return
        if length < 1 or length > MAX_BODY:
            self.error(413, "Body must contain 1 through 8192 bytes")
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete body")
            body = json.loads(raw, object_pairs_hook=unique_object)
        except (ValueError, UnicodeDecodeError):
            self.error(400, "A complete JSON object with unique keys is required")
            return
        except TimeoutError:
            self.error(408, "Timed out reading request body")
            return
        if not isinstance(body, dict):
            self.error(400, "A JSON object is required")
            return
        if not isinstance(body.get("model"), str):
            self.error(400, "A string model is required")
            return
        if body["model"] != self.server.model:
            self.error(404, "Model is not served by this replica")
            return
        if body.get("stream", False) is not False:
            self.error(400, "This mock supports only non-streaming requests")
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict)
            or message.get("role") not in ("system", "developer", "user", "assistant")
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            self.error(400, "messages must be a nonempty list of text chat messages")
            return
        self.send_json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.server.model,
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": "MOCK response: no inference was performed."},
                         "finish_reason": "stop"}],
        }, replica=True)


class InferenceServer(ThreadingHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--replica", required=True)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    with InferenceServer(("127.0.0.1", args.port), InferenceHandler) as server:
        server.model = args.model
        server.replica = args.replica
        print(f"MOCK {args.replica}: {args.model} on 127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
