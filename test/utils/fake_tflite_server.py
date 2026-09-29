#!/usr/bin/env python3
"""
Stand-in for tflite-server's image-classification contract, for exercising
lemond's /v1/images/classify forwarding without a LiteRT build.

Install it on PATH as `tflite-server` (see server_image_classify.py). It
accepts the flags lemond passes, answers GET /health with
task=image-classification, and serves POST /classify/image:

- a body containing POISON crashes the process with SIGSEGV, which is how the
  non-replay test proves lemond does not resend a crashing input;
- a PNG whose IHDR claims more than --max-image-pixels pixels, or whose data
  is truncated, gets 400 like the real decoder;
- anything else gets a fixed ranked prediction list.

When FAKE_TFLITE_STATE_DIR is set, each start and each crash appends a line to
starts.log / crashes.log there.
"""

import argparse
import json
import os
import signal
import struct
import sys
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

POISON = b"B169-POISON-IMAGE"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _log(name, line):
    state_dir = os.environ.get("FAKE_TFLITE_STATE_DIR")
    if state_dir:
        with open(os.path.join(state_dir, name), "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _png_problem(data, max_pixels):
    """Returns an error string for a PNG the real decoder would refuse."""
    if len(data) < 33 or data[12:16] != b"IHDR":
        return "corrupt or truncated image"
    width, height = struct.unpack(">II", data[16:24])
    if width * height > max_pixels:
        return f"image has {width * height} pixels, over the {max_pixels} limit"
    if b"IEND" not in data:
        return "corrupt or truncated image"
    return None


def _image_part(handler, body):
    content_type = handler.headers.get("Content-Type", "")
    message = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    image, top_k = None, 5
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name in ("image", "file"):
            image = part.get_payload(decode=True)
        elif name == "top_k":
            top_k = int(part.get_payload(decode=True).decode())
    return image, top_k


def make_handler(args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self._send(
                    200,
                    {"status": "ok", "engine": "fake", "task": "image-classification"},
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/classify":
                self._send(
                    400,
                    {
                        "error": "this server hosts an image-classification model; "
                        "POST /classify/image"
                    },
                )
                return
            if self.path != "/classify/image":
                self._send(404, {"error": "not found"})
                return
            image, top_k = _image_part(self, body)
            if image is None:
                self._send(400, {"error": "exactly one image part required"})
                return
            if POISON in image:
                _log("crashes.log", str(os.getpid()))
                os.kill(os.getpid(), signal.SIGSEGV)
            if image.startswith(PNG_SIGNATURE):
                problem = _png_problem(image, args.max_image_pixels)
                if problem:
                    self._send(400, {"error": problem})
                    return
            ranked = [
                {"index": 653, "label": "military uniform", "score": 0.80},
                {"index": 440, "label": "bearskin", "score": 0.04},
                {"index": 668, "label": "mortarboard", "score": 0.02},
                {"index": 135, "label": "crane", "score": 0.01},
                {"index": 518, "label": "crane", "score": 0.005},
            ][: max(1, top_k)]
            self._send(
                200,
                {
                    "predictions": ranked,
                    "labels": {p["label"]: p["score"] for p in ranked},
                    "timings": {"decode_ms": 0.1, "inference_ms": 0.1},
                },
            )

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--weight-cache")
    parser.add_argument("--max-image-pixels", type=int, default=4_000_000)
    args, _ = parser.parse_known_args()
    manifest = os.path.join(args.model_path, "manifest.json")
    if not os.path.exists(manifest):
        print("tflite-server: fake supports image models only", file=sys.stderr)
        sys.exit(1)
    _log("starts.log", str(os.getpid()))
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args)).serve_forever()


if __name__ == "__main__":
    main()
