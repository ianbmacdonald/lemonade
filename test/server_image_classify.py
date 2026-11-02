"""
Image-classification tests for Lemonade Server: POST /v1/images/classify with
the tflite (tflite-server >= 0.2.0), executorch (et-server) or onnxruntime
(ort-server) recipe.

Usage:
    python server_image_classify.py --wrapped-server tflite --backend system \\
        --image-model-dir <dir> --hf-cache <lemond's HF cache dir>

    --wrapped-server picks the recipe: tflite (the default), executorch or
    onnxruntime. <dir> is an image model directory for that engine (the graph,
    model.tflite, model.pte, or model.onnx or model.ort, plus labels.txt and
    manifest.json), such as the one tflite-server's tools/make_image_model_dir.py
    writes for TF MobileNetV2 1.0 224. It is copied into the server's HF cache
    and registered with a local import as user.MobileNetV2-1.0-224-<engine>, so
    the server must run on this machine. LEMONADE_TEST_IMAGE_MODEL_DIR and
    LEMONADE_TEST_HF_CACHE work in place of the flags. tflite-server and
    et-server are found on PATH; lemond finds ort-server through its install or
    `onnxruntime.cpu_bin`, so that check is left to the load.

    --fake-backend runs the same suite against test/utils/fake_tflite_server.py
    installed on the server's PATH as `tflite-server` or `et-server` (for
    et-server, started with `--engine et-server`), or set as
    `onnxruntime.cpu_bin` (started with `--engine ort-server`). It skips the
    accuracy assertions and enables the crash non-replay test, which needs a
    backend that crashes on demand; pass --fake-state-dir, the
    FAKE_TFLITE_STATE_DIR the server was started with.

    --short-watchdog says the server was started with
    LEMONADE_BACKEND_WATCHDOG_GRACE_SECONDS=10 and
    LEMONADE_BACKEND_WATCHDOG_POLL_SECONDS=1, which enables the test of an input
    that hangs the engine without crashing it.

    --lemond-pid enables the resident-memory assertion on the oversized upload.

Negative tests need no model and run first. Positive tests skip, with the
reason, when no model directory or no engine is available.

The fixture test_image_grace_hopper.jpg is the US Navy portrait of Grace Hopper
(public domain; sha256 a8ca6d73...7130), as used in the TensorFlow Lite
examples. MobileNetV2's reference answer is class 653 "military uniform" at
0.803491 (PIL bilinear stretch to 224, x/127.5 - 1).
"""

import argparse
import base64
import json
import os
import shutil
import gzip
import socket
import struct
import threading
import time
import zlib

import requests

from utils.server_base import (
    ServerTestBase,
    run_server_tests,
    load_model,
    unload_model,
)
from utils.test_models import PORT, TIMEOUT_DEFAULT

TIMEOUT_CLASSIFY = 300

ENGINES = {
    "tflite": {
        "graph": "model.tflite",
        "binary": "tflite-server",
        "args": "tflite_args",
        "image_release": "tflite-server 0.2.0",
        "model": "user.MobileNetV2-1.0-224-TFLite",
        "text_model": "user.B194-Text-TFLite",
    },
    "executorch": {
        "graph": "model.pte",
        "binary": "et-server",
        "args": "executorch_args",
        "image_release": "et-server 0.2.0",
        "model": "user.MobileNetV2-1.0-224-ExecuTorch",
        "text_model": "user.B204-Text-ExecuTorch",
    },
    "onnxruntime": {
        "graph": "model.onnx",
        "binary": None,
        "args": "onnxruntime_args",
        "image_release": "ort-server 0.2.0-musl",
        "model": "user.MobileNetV2-1.0-224-ONNX",
        "text_model": "user.B204-Text-ONNX",
    },
}
RECIPE = "tflite"
ENGINE = ENGINES[RECIPE]
MODEL = ENGINE["model"]
TEXT_MODEL = ENGINE["text_model"]
POISON_TEXT = "B194-POISON-TEXT"
HANG_TEXT = "B194-HANG-TEXT"
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_image_grace_hopper.jpg"
)
REFERENCE_INDEX = 653
REFERENCE_SCORE = 0.803491
REFERENCE_TOL = 0.02
POISON = b"B169-POISON-IMAGE"
SLOW = b"B169-SLOW-IMAGE"
HANG_IMAGE = b"B194-HANG-IMAGE"

_opts = argparse.Namespace(
    image_model_dir=None,
    hf_cache=None,
    fake_backend=False,
    fake_state_dir=None,
    lemond_pid=None,
    short_watchdog=False,
    fake_ort=False,
)


def _parse_own_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--image-model-dir", default=os.environ.get("LEMONADE_TEST_IMAGE_MODEL_DIR")
    )
    parser.add_argument("--hf-cache", default=os.environ.get("LEMONADE_TEST_HF_CACHE"))
    parser.add_argument("--fake-backend", action="store_true")
    parser.add_argument(
        "--fake-state-dir", default=os.environ.get("FAKE_TFLITE_STATE_DIR")
    )
    parser.add_argument("--lemond-pid", type=int)
    parser.add_argument("--short-watchdog", action="store_true")
    # lemond was started with onnxruntime.cpu_bin pointing at the fake engine.
    parser.add_argument("--fake-ort", action="store_true")
    parser.add_argument("--wrapped-server", default="tflite", choices=sorted(ENGINES))
    args, _ = parser.parse_known_args()
    return args


def _png(width, height, truncate=False):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\x80\x40\x20" * width for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return png[: len(png) // 2] if truncate else png


def _jpeg_like(payload=b""):
    return b"\xff\xd8\xff\xe0" + payload + b"\x00" * 64


def _raw_request(headers, body=b""):
    """Sends a request without a client library, so headers such as a
    Content-Length far larger than the body reach the server unaltered."""
    with socket.create_connection(("localhost", PORT), timeout=30) as sock:
        try:
            sock.sendall(headers.encode() + body)
        except (BrokenPipeError, ConnectionResetError):
            # The server may answer and close before a large body is sent.
            pass
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        length = 0
        for line in head.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value.strip())
        while len(rest) < length:
            chunk = sock.recv(65536)
            if not chunk:
                break
            rest += chunk
        return status, rest


def _write_stub_model_dir(path):
    """An image model dir for the fake backend, which never reads the model."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, ENGINE["graph"]), "wb") as f:
        f.write(b"stub")
    with open(os.path.join(path, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("background\nmilitary uniform\n")
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"task": "image-classification", "labels_file": "labels.txt"}, f)
    return os.path.abspath(path)


def _write_stub_text_model_dir(path):
    """A text model dir for the fake backend, which never reads the model."""
    os.makedirs(path, exist_ok=True)
    for name, content in (
        (ENGINE["graph"], "stub"),
        ("tokenizer.json", "{}"),
        ("config.json", "{}"),
    ):
        with open(os.path.join(path, name), "w", encoding="utf-8") as f:
            f.write(content)
    return os.path.abspath(path)


def _engine_args(value):
    return {ENGINE["args"]: value}


def _line_count(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return len(f.read().splitlines())


def _vm_rss_kb(pid):
    with open(f"/proc/{pid}/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


class ImageClassifyTests(ServerTestBase):
    """Tests for the /images/classify endpoint."""

    _registered = False
    _text_registered = False

    def _url(self, prefix="/api/v1"):
        return f"http://localhost:{PORT}{prefix}/images/classify"

    def _post_json(self, payload, prefix="/api/v1"):
        return requests.post(self._url(prefix), json=payload, timeout=TIMEOUT_CLASSIFY)

    def _post_form(self, files, data=None, prefix="/api/v1"):
        return requests.post(
            self._url(prefix), files=files, data=data or {}, timeout=TIMEOUT_CLASSIFY
        )

    def _assert_status(self, response, expected, context):
        self.assertEqual(
            response.status_code,
            expected,
            f"{context}: expected {expected}, got {response.status_code}: {response.text[:500]}",
        )
        if expected != 200:
            self.assertIn("error", response.json(), f"{context}: missing 'error'")
        print(f"[OK] {context}: {response.status_code}")

    # ---- negative tests: no model needed ----

    def test_001_get_is_405(self):
        response = requests.get(self._url(), timeout=TIMEOUT_DEFAULT)
        self.assertEqual(response.status_code, 405)

    def test_002_text_plain_rejected(self):
        response = requests.post(
            self._url(),
            data="hello",
            headers={"Content-Type": "text/plain"},
            timeout=TIMEOUT_DEFAULT,
        )
        self._assert_status(response, 400, "text/plain body")

    def test_003_malformed_json_inputs_rejected(self):
        jpeg = base64.b64encode(_jpeg_like()).decode()
        cases = [
            ({"image": "@@@@"}, "invalid base64"),
            (
                {"image": base64.b64encode(b"GIF89a-not-supported").decode()},
                "GIF bytes",
            ),
            ({"image": "https://example.com/cat.jpg"}, "remote URL"),
            ({"image": jpeg, "top_k": 0}, "top_k=0"),
            ({"image": jpeg, "top_k": 2.5}, "float top_k"),
            ({"image": "data:text/plain;base64,aGVsbG8="}, "non-image data URL"),
            ({}, "missing image"),
        ]
        for payload, context in cases:
            self._assert_status(self._post_json(payload), 400, context)

    def test_004_decoded_image_over_16_mib_is_413(self):
        big = _jpeg_like(b"\x00" * (17 * 1024 * 1024))
        response = self._post_json({"image": base64.b64encode(big).decode()})
        self._assert_status(response, 413, "17 MiB decoded image")

    def test_005_oversized_body_refused_before_it_is_read(self):
        rss_before = _vm_rss_kb(_opts.lemond_pid) if _opts.lemond_pid else None
        # Only the headers are sent. A server that waited for the declared 50 MB
        # body would never answer, so an answer at all shows the body was not read.
        status, body = _raw_request(
            f"POST /api/v1/images/classify HTTP/1.1\r\nHost: localhost:{PORT}\r\n"
            "Content-Type: application/json\r\nContent-Length: 50000000\r\n\r\n"
        )
        self.assertEqual(status, 413, body[:300])
        self.assertEqual(json.loads(body)["error"]["code"], "payload_too_large")
        if rss_before is not None:
            growth_kb = _vm_rss_kb(_opts.lemond_pid) - rss_before
            self.assertLess(growth_kb, 10 * 1024, f"lemond RSS grew {growth_kb} KiB")
            print(f"[OK] lemond RSS growth {growth_kb} KiB")

    def test_006_chunked_upload_is_411(self):
        status, body = _raw_request(
            f"POST /api/v1/images/classify HTTP/1.1\r\nHost: localhost:{PORT}\r\n"
            "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n",
            b"5\r\nhello\r\n0\r\n\r\n",
        )
        self.assertEqual(status, 411, body[:300])
        self.assertEqual(json.loads(body)["error"]["code"], "length_required")

    def _assert_refused_unread(self, headers, body, expected_status, expected_code):
        rss_before = _vm_rss_kb(_opts.lemond_pid) if _opts.lemond_pid else None
        status, reply = _raw_request(headers, body)
        self.assertEqual(status, expected_status, reply[:300])
        self.assertEqual(json.loads(reply)["error"]["code"], expected_code)
        if rss_before is not None:
            growth_kb = _vm_rss_kb(_opts.lemond_pid) - rss_before
            self.assertLess(growth_kb, 10 * 1024, f"lemond RSS grew {growth_kb} KiB")
            print(f"[OK] lemond RSS growth {growth_kb} KiB")

    def test_006b_gzip_body_is_415(self):
        # 60 MiB of JSON that gzips to about 60 KB: httplib would inflate it
        # past the 61 KB Content-Length the guard saw.
        payload = gzip.compress(
            b'{"image":"' + b"A" * (60 * 1024 * 1024) + b'"}', compresslevel=9
        )
        self._assert_refused_unread(
            f"POST /api/v1/images/classify HTTP/1.1\r\nHost: localhost:{PORT}\r\n"
            "Content-Type: application/json\r\nContent-Encoding: gzip\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n",
            payload,
            415,
            "unsupported_content_encoding",
        )

    def test_006c_chunked_with_zero_content_length_is_411(self):
        # A small first slice of a chunked body; httplib would read chunks
        # until the terminator, however many the client sends.
        chunk = b"A" * (32 * 1024)
        body = b"".join(b"%x\r\n" % len(chunk) + chunk + b"\r\n" for _ in range(2))
        self._assert_refused_unread(
            f"POST /api/v1/images/classify HTTP/1.1\r\nHost: localhost:{PORT}\r\n"
            "Content-Type: application/json\r\nContent-Length: 0\r\n"
            "Transfer-Encoding: chunked\r\n\r\n",
            body,
            411,
            "length_required",
        )

    def test_006d_chunked_with_small_content_length_refused(self):
        # httplib itself refuses a positive Content-Length with
        # Transfer-Encoding as request smuggling (400) before lemond's guard
        # runs; either refusal leaves the body unread.
        rss_before = _vm_rss_kb(_opts.lemond_pid) if _opts.lemond_pid else None
        status, reply = _raw_request(
            f"POST /api/v1/images/classify HTTP/1.1\r\nHost: localhost:{PORT}\r\n"
            "Content-Type: application/json\r\nContent-Length: 1024\r\n"
            "Transfer-Encoding: chunked\r\n\r\n",
            b"5\r\nhello\r\n0\r\n\r\n",
        )
        self.assertIn(status, (400, 411), reply[:300])
        if rss_before is not None:
            growth_kb = _vm_rss_kb(_opts.lemond_pid) - rss_before
            self.assertLess(growth_kb, 10 * 1024, f"lemond RSS grew {growth_kb} KiB")
        print(f"[OK] Content-Length + Transfer-Encoding refused: {status}")

    def test_006e_aborted_uploads_release_their_slot(self):
        # Each upload is admitted, then the client disappears mid-body. If the
        # in-flight slot leaked, a few of these would leave the route at 503.
        for _ in range(4):
            with socket.create_connection(("localhost", PORT), timeout=30) as sock:
                sock.sendall(
                    (
                        f"POST /api/v1/images/classify HTTP/1.1\r\nHost: localhost:{PORT}\r\n"
                        "Content-Type: application/json\r\nContent-Length: 100000\r\n\r\n"
                    ).encode()
                    + b'{"image":"'
                )
        time.sleep(1)
        self._assert_status(
            self._post_json({"image": "@@@@"}), 400, "request after aborted uploads"
        )

    def test_007_two_image_parts_rejected(self):
        files = [
            ("image", ("a.jpg", _jpeg_like(), "image/jpeg")),
            ("file", ("b.jpg", _jpeg_like(), "image/jpeg")),
        ]
        self._assert_status(self._post_form(files), 400, "image + file parts")
        files = [
            ("image", ("a.jpg", _jpeg_like(), "image/jpeg")),
            ("image", ("b.jpg", _jpeg_like(), "image/jpeg")),
        ]
        self._assert_status(self._post_form(files), 400, "two image parts")
        self._assert_status(
            self._post_form({"other": ("a.jpg", _jpeg_like())}), 400, "no parts"
        )

    def test_008_text_classifier_model_rejected(self):
        # Registration only; nothing is downloaded or loaded.
        name = "user.B169-Text-Classifier-Probe"
        response = requests.post(
            f"{self.base_url}/models/register",
            json={
                "model_name": name,
                "checkpoint": "example/text-classifier-probe",
                "recipe": "onnxruntime",
                "labels": ["classification"],
            },
            timeout=TIMEOUT_DEFAULT,
        )
        self.assertEqual(response.status_code, 200, response.text[:500])
        response = self._post_json(
            {"model": name, "image": base64.b64encode(_jpeg_like()).decode()}
        )
        self._assert_status(response, 400, "text classifier on /images/classify")
        self.assertEqual(response.json()["error"].get("code"), "model_not_applicable")

    def test_009_no_model_and_none_loaded(self):
        response = self._post_form({"image": ("a.jpg", _jpeg_like(), "image/jpeg")})
        self._assert_status(response, 400, "implicit model with none loaded")

    # ---- positive tests: need the model ----

    def _require_model(self):
        if _opts.fake_backend and not _opts.image_model_dir and _opts.hf_cache:
            _opts.image_model_dir = _write_stub_model_dir(
                os.path.join(_opts.hf_cache, "..", f"b169-stub-image-model-{RECIPE}")
            )
        if not _opts.image_model_dir or not os.path.isdir(_opts.image_model_dir):
            self.skipTest("no --image-model-dir (or LEMONADE_TEST_IMAGE_MODEL_DIR)")
        if not _opts.hf_cache:
            self.skipTest("no --hf-cache (or LEMONADE_TEST_HF_CACHE)")
        if ENGINE["binary"] and shutil.which(ENGINE["binary"]) is None:
            self.skipTest(f"{ENGINE['binary']} is not on PATH")
        if ImageClassifyTests._registered:
            return
        dest = os.path.join(_opts.hf_cache, "models--" + MODEL[len("user.") :])
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(_opts.image_model_dir, dest)
        response = requests.post(
            f"{self.base_url}/pull",
            json={
                "model_name": MODEL,
                "recipe": RECIPE,
                "labels": ["image-classification"],
                "local_import": True,
            },
            timeout=TIMEOUT_DEFAULT,
        )
        self.assertEqual(response.status_code, 200, response.text[:500])
        ImageClassifyTests._registered = True

    def _assert_reference(self, response, top_k):
        self._assert_status(response, 200, "classification")
        body = response.json()
        self.assertEqual(body["object"], "image_classification")
        # The public id of a user model drops the "user." prefix when echoed
        # for an implicitly selected model.
        self.assertIn(body["model"], (MODEL, MODEL[len("user.") :]))
        data = body["data"]
        self.assertEqual(len(data), top_k)
        scores = [d["score"] for d in data]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIsInstance(body["labels"], dict)
        if not _opts.fake_backend:
            self.assertEqual(data[0]["index"], REFERENCE_INDEX, data[0])
            self.assertAlmostEqual(
                data[0]["score"], REFERENCE_SCORE, delta=REFERENCE_TOL
            )
        print(f"[OK] top-1 {data[0]}")

    def _fixture(self):
        with open(FIXTURE, "rb") as f:
            return f.read()

    def test_500_multipart(self):
        self._require_model()
        response = self._post_form(
            {"image": ("grace_hopper.jpg", self._fixture(), "image/jpeg")},
            data={"model": MODEL, "top_k": "3"},
        )
        self._assert_reference(response, 3)

    def test_501_json_base64(self):
        self._require_model()
        image = base64.b64encode(self._fixture()).decode()
        self._assert_reference(
            self._post_json({"model": MODEL, "image": image, "top_k": 5}), 5
        )

    def test_502_json_data_url_and_file_part(self):
        self._require_model()
        image = "data:image/jpeg;base64," + base64.b64encode(self._fixture()).decode()
        self._assert_reference(
            self._post_json({"model": MODEL, "image": image, "top_k": 2}), 2
        )
        response = self._post_form(
            {"file": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"model": MODEL, "top_k": "1"},
        )
        self._assert_reference(response, 1)

    def test_503_implicit_model(self):
        self._require_model()
        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")}, data={"top_k": "3"}
        )
        self._assert_reference(response, 3)

    def test_504_text_route_refuses_image_model(self):
        self._require_model()
        response = requests.post(
            f"{self.base_url}/classify",
            json={"model": MODEL, "input": "hello"},
            timeout=TIMEOUT_DEFAULT,
        )
        self._assert_status(response, 400, "/classify on an image model")

    def test_505_all_prefixes(self):
        self._require_model()
        for prefix in ("/api/v0", "/api/v1", "/v0", "/v1"):
            response = self._post_form(
                {"image": ("g.jpg", self._fixture(), "image/jpeg")},
                data={"model": MODEL, "top_k": "1"},
                prefix=prefix,
            )
            self._assert_reference(response, 1)

    def test_506_backend_400_preserved_for_truncated_png(self):
        self._require_model()
        response = self._post_form(
            {"image": ("t.png", _png(64, 64, truncate=True), "image/png")},
            data={"model": MODEL},
        )
        self._assert_status(response, 400, "truncated PNG")

    def test_507_backend_400_preserved_for_pixel_cap(self):
        self._require_model()
        unload_model(MODEL)
        load_model(MODEL, **_engine_args("--max-image-pixels 1000"))
        try:
            response = self._post_form(
                {"image": ("p.png", _png(64, 64), "image/png")},
                data={"model": MODEL},
            )
            self._assert_status(response, 400, "64x64 PNG over a 1000-pixel cap")
        finally:
            unload_model(MODEL)
            load_model(MODEL, **_engine_args(""))

    def test_507b_load_refuses_bind_override(self):
        # The *_args option comes from the request body; --host would put the
        # engine on the LAN without lemond's API key.
        self._require_model()
        unload_model(MODEL)
        refused = ["--host 0.0.0.0", "--host=0.0.0.0"]
        if RECIPE != "onnxruntime":
            refused.append("--weight-cache /tmp/x")
        try:
            for args in refused:
                response = load_model(MODEL, **_engine_args(args))
                self.assertEqual(response.status_code, 400, args)
                self.assertEqual(
                    response.json()["error"]["code"], "invalid_backend_args", args
                )
                self.assertIn("managed by Lemonade", response.text, args)
        finally:
            load_model(MODEL, **_engine_args(""))

    def test_507b2_load_refuses_a_nul_spliced_flag(self):
        # The engine's argv ends the first token at the NUL, leaving --host.
        self._require_model()
        unload_model(MODEL)
        try:
            response = load_model(MODEL, **_engine_args("--host\x00x 0.0.0.0"))
            self.assertEqual(response.status_code, 400, response.text[:500])
            self.assertIn("NUL", response.text)
        finally:
            load_model(MODEL, **_engine_args(""))

    def test_507c_weight_cache_symlink_not_used(self):
        # The engine would write its weight cache through the link, which in an
        # HF snapshot points into the shared blob store.
        if RECIPE != "tflite":
            self.skipTest("only tflite-server takes a weight cache from lemond")
        if not _opts.fake_backend or not _opts.fake_state_dir:
            self.skipTest("needs --fake-backend and --fake-state-dir")
        self._require_model()
        args_log = os.path.join(_opts.fake_state_dir, "args.log")
        cache = os.path.join(
            _opts.hf_cache,
            "models--" + MODEL[len("user.") :],
            "model.tflite.xnnpack_cache",
        )
        outside = os.path.join(_opts.fake_state_dir, "outside-cache")

        def last_args():
            with open(args_log, encoding="utf-8") as f:
                return json.loads(f.read().splitlines()[-1])

        try:
            if os.path.lexists(cache):
                os.remove(cache)
            os.symlink(outside, cache)
            unload_model(MODEL)
            self.assertEqual(load_model(MODEL).status_code, 200)
            self.assertNotIn("--weight-cache", last_args())
        finally:
            if os.path.lexists(cache):
                os.remove(cache)
        unload_model(MODEL)
        self.assertEqual(load_model(MODEL).status_code, 200)
        self.assertIn(cache, last_args())

    def test_508_crash_is_not_replayed(self):
        if not _opts.fake_backend or not _opts.fake_state_dir:
            self.skipTest("needs --fake-backend and --fake-state-dir")
        self._require_model()
        crashes = os.path.join(_opts.fake_state_dir, "crashes.log")
        starts = os.path.join(_opts.fake_state_dir, "starts.log")

        def lines(path):
            if not os.path.exists(path):
                return 0
            with open(path, encoding="utf-8") as f:
                return len(f.read().splitlines())

        # Make sure the model is up so the baseline counts are settled.
        self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")}, {"model": MODEL}
        )
        crashes_before, starts_before = lines(crashes), lines(starts)

        response = self._post_form(
            {"image": ("poison.jpg", _jpeg_like(POISON), "image/jpeg")},
            data={"model": MODEL},
        )
        self._assert_status(response, 502, "poison image")
        self.assertEqual(response.json()["error"]["code"], "backend_crashed_on_input")
        self.assertEqual(
            lines(crashes) - crashes_before, 1, "the poison image was replayed"
        )

        # No model named: the sole image model is chosen even though its
        # backend is down, and this request reloads it.
        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"top_k": "1"},
        )
        self._assert_reference(response, 1)
        self.assertEqual(
            lines(starts) - starts_before, 1, "expected exactly one reload"
        )

    def test_509_inflight_cap(self):
        if not _opts.fake_backend:
            self.skipTest("needs --fake-backend")
        self._require_model()
        results = []

        def slow_request():
            results.append(
                self._post_form(
                    {"image": ("slow.jpg", _jpeg_like(SLOW), "image/jpeg")},
                    data={"model": MODEL, "top_k": "1"},
                )
            )

        threads = [threading.Thread(target=slow_request) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(1.5)
        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"model": MODEL, "top_k": "1"},
        )
        for t in threads:
            t.join()
        self._assert_status(response, 503, "third concurrent request")
        self.assertEqual(response.json()["error"]["code"], "server_busy")
        self.assertEqual(response.headers.get("Retry-After"), "1")
        for r in results:
            self._assert_reference(r, 1)
        self._assert_reference(
            self._post_form(
                {"image": ("g.jpg", self._fixture(), "image/jpeg")},
                data={"model": MODEL, "top_k": "1"},
            ),
            1,
        )

    def test_510_repeated_crashes_pause_the_model(self):
        # Runs last: it leaves the model paused for up to a minute.
        if not _opts.fake_backend or not _opts.fake_state_dir:
            self.skipTest("needs --fake-backend and --fake-state-dir")
        self._require_model()
        crashes = os.path.join(_opts.fake_state_dir, "crashes.log")
        starts = os.path.join(_opts.fake_state_dir, "starts.log")

        def lines(path):
            if not os.path.exists(path):
                return 0
            with open(path, encoding="utf-8") as f:
                return len(f.read().splitlines())

        def poison():
            return self._post_form(
                {"image": ("poison.jpg", _jpeg_like(POISON), "image/jpeg")},
                data={"model": MODEL},
            )

        response = None
        for _ in range(4):
            response = poison()
            if response.status_code != 502:
                break
        self._assert_status(response, 503, "poison after repeated crashes")
        self.assertEqual(response.json()["error"]["code"], "backend_crash_cooldown")
        retry_after = int(response.headers["Retry-After"])
        self.assertTrue(1 <= retry_after <= 60, retry_after)

        crashes_before, starts_before = lines(crashes), lines(starts)
        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"model": MODEL, "top_k": "1"},
        )
        self._assert_status(response, 503, "good image while paused")
        implicit = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")}, data={"top_k": "1"}
        )
        self._assert_status(implicit, 503, "implicit model while paused")
        self.assertEqual(lines(crashes), crashes_before, "paused model crashed again")
        self.assertEqual(lines(starts), starts_before, "paused model was reloaded")

        time.sleep(int(response.headers["Retry-After"]) + 1)
        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"model": MODEL, "top_k": "1"},
        )
        self._assert_reference(response, 1)
        self.assertEqual(
            lines(starts) - starts_before, 1, "expected one reload after the pause"
        )

    # ---- text classification on the same engine (fake backend only) ----

    def _require_text_model(self):
        if not _opts.fake_backend or not _opts.fake_state_dir or not _opts.hf_cache:
            self.skipTest("needs --fake-backend, --fake-state-dir and --hf-cache")
        if ImageClassifyTests._text_registered:
            return
        dest = os.path.join(_opts.hf_cache, "models--" + TEXT_MODEL[len("user.") :])
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        _write_stub_text_model_dir(dest)
        response = requests.post(
            f"{self.base_url}/pull",
            json={
                "model_name": TEXT_MODEL,
                "recipe": RECIPE,
                "labels": ["classification"],
                "local_import": True,
            },
            timeout=TIMEOUT_DEFAULT,
        )
        self.assertEqual(response.status_code, 200, response.text[:500])
        ImageClassifyTests._text_registered = True

    def _classify_text(self, text, **extra):
        return requests.post(
            f"{self.base_url}/classify",
            json={"model": TEXT_MODEL, "input": text, **extra},
            timeout=TIMEOUT_CLASSIFY,
        )

    def test_511_text_labels_refused_by_fixed_label_model(self):
        self._require_text_model()
        self._assert_status(self._classify_text("hello"), 200, "text classify")
        if RECIPE == "onnxruntime":
            self.skipTest("ort-server takes caller labels for zero-shot models")
        response = self._classify_text("hello", labels=["caller-category"])
        self._assert_status(response, 400, "labels sent to a fixed-label model")

    def test_512_text_crash_is_not_replayed(self):
        self._require_text_model()
        crashes = os.path.join(_opts.fake_state_dir, "crashes.log")
        starts = os.path.join(_opts.fake_state_dir, "starts.log")
        self._assert_status(self._classify_text("hello"), 200, "warm-up")
        crashes_before, starts_before = _line_count(crashes), _line_count(starts)

        response = self._classify_text(POISON_TEXT)
        self._assert_status(response, 502, "poison text")
        self.assertEqual(response.json()["error"]["code"], "backend_crashed_on_input")
        self.assertEqual(
            _line_count(crashes) - crashes_before, 1, "the poison text was replayed"
        )

        self._assert_status(self._classify_text("hello"), 200, "after the crash")
        self.assertEqual(
            _line_count(starts) - starts_before, 1, "expected exactly one reload"
        )

    def test_512b_omitted_text_model_restarts_after_a_crash(self):
        self._require_text_model()
        starts = os.path.join(_opts.fake_state_dir, "starts.log")
        self._assert_status(self._classify_text("hello"), 200, "warm-up")
        self._assert_status(self._classify_text(POISON_TEXT), 502, "poison text")
        starts_before = _line_count(starts)

        # No model named: the sole classifier is chosen even though its
        # backend is down, and this request restarts it.
        response = requests.post(
            f"{self.base_url}/classify",
            json={"input": "hello"},
            timeout=TIMEOUT_CLASSIFY,
        )
        self._assert_status(response, 200, "omitted model after a crash")
        self.assertEqual(
            _line_count(starts) - starts_before, 1, "expected exactly one restart"
        )

    def test_512c_unresponsive_text_is_not_counted_as_a_crash(self):
        if not _opts.short_watchdog:
            self.skipTest("needs --short-watchdog")
        self._require_text_model()
        hangs = os.path.join(_opts.fake_state_dir, "hangs.log")
        self._assert_status(self._classify_text("hello"), 200, "warm-up")
        for attempt in range(3):
            hangs_before = _line_count(hangs)
            response = self._classify_text(HANG_TEXT)
            self._assert_status(response, 504, f"hanging text {attempt + 1}")
            self.assertEqual(
                response.json()["error"]["code"], "backend_unresponsive_on_input"
            )
            self.assertEqual(
                _line_count(hangs) - hangs_before, 1, "the hanging text was replayed"
            )
        # Three slow inputs are not three crashes, so the model is not paused.
        self._assert_status(self._classify_text("hello"), 200, "after three hangs")

    def test_512d_hanging_image_resets_the_backend(self):
        # Under the default watchdog (about 105 s to a kill) the image forward
        # must outlast the kill, or a hung engine is never reset.
        if _opts.short_watchdog:
            self.skipTest("needs the default watchdog")
        if not _opts.fake_backend or not _opts.fake_state_dir:
            self.skipTest("needs --fake-backend and --fake-state-dir")
        self._require_model()
        hangs = os.path.join(_opts.fake_state_dir, "hangs.log")
        self._assert_reference(
            self._post_form(
                {"image": ("g.jpg", self._fixture(), "image/jpeg")},
                data={"model": MODEL, "top_k": "1"},
            ),
            1,
        )
        hangs_before = _line_count(hangs)
        response = self._post_form(
            {"image": ("hang.jpg", _jpeg_like(HANG_IMAGE), "image/jpeg")},
            data={"model": MODEL},
        )
        self._assert_status(response, 504, "hanging image")
        self.assertEqual(
            response.json()["error"]["code"], "backend_unresponsive_on_input"
        )
        self.assertEqual(_line_count(hangs) - hangs_before, 1)
        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"model": MODEL, "top_k": "1"},
        )
        self._assert_reference(response, 1)

    def _register_stub_ort(self, name, fail_start):
        dest = os.path.join(_opts.hf_cache, "models--" + name[len("user.") :])
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(dest)
        files = ["model.onnx", "tokenizer.json", "config.json"]
        if fail_start:
            files.append("FAKE_FAIL_START")
        for file_name in files:
            with open(os.path.join(dest, file_name), "w", encoding="utf-8") as f:
                f.write("{}")
        response = requests.post(
            f"{self.base_url}/pull",
            json={
                "model_name": name,
                "recipe": "onnxruntime",
                "labels": ["classification"],
                "local_import": True,
            },
            timeout=TIMEOUT_DEFAULT,
        )
        self.assertEqual(response.status_code, 200, response.text[:500])
        return name

    def _assert_image_model_resident(self):
        health = requests.get(f"{self.base_url}/health", timeout=TIMEOUT_DEFAULT).json()
        # /health lists public names, without the user. prefix.
        loaded = [m["model_name"] for m in health["all_models_loaded"]]
        self.assertIn(MODEL[len("user.") :], loaded, "the image model was evicted")

    def test_515_broken_ort_model_keeps_resident_models(self):
        if not _opts.fake_ort:
            self.skipTest("needs --fake-ort")
        self._require_model()
        failed = os.path.join(_opts.fake_state_dir, "failed-starts.log")
        self._assert_reference(
            self._post_form(
                {"image": ("g.jpg", self._fixture(), "image/jpeg")},
                data={"model": MODEL, "top_k": "1"},
            ),
            1,
        )
        name = self._register_stub_ort("user.B194-Broken-Ort", fail_start=True)
        failed_before = _line_count(failed)
        response = requests.post(
            f"{self.base_url}/load", json={"model_name": name}, timeout=TIMEOUT_DEFAULT
        )
        self.assertGreaterEqual(response.status_code, 400, response.text[:500])
        # One start and one relaunch to capture its stderr; evicting the other
        # models and loading again cannot fix a CPU engine that refused its model.
        self.assertEqual(_line_count(failed) - failed_before, 2)
        self._assert_image_model_resident()

    def test_515b_missing_ort_engine_keeps_resident_models(self):
        # Without --fake-ort the server runs offline with no ort-server
        # installed, so the load fails at the install step.
        if _opts.fake_ort or not _opts.fake_backend:
            self.skipTest("needs --fake-backend without --fake-ort")
        self._require_model()
        self._assert_reference(
            self._post_form(
                {"image": ("g.jpg", self._fixture(), "image/jpeg")},
                data={"model": MODEL, "top_k": "1"},
            ),
            1,
        )
        name = self._register_stub_ort("user.B194-Missing-Ort", fail_start=False)
        response = requests.post(
            f"{self.base_url}/load", json={"model_name": name}, timeout=TIMEOUT_DEFAULT
        )
        self.assertGreaterEqual(response.status_code, 400, response.text[:500])
        self._assert_image_model_resident()

    def test_516_label_and_layout_must_agree(self):
        # Runs last: loading a third model of a type can evict the first.
        if not _opts.hf_cache:
            self.skipTest("no --hf-cache (or LEMONADE_TEST_HF_CACHE)")
        cases = (
            ("user.B204-Image-Label-On-Text-Dir", "image-classification", True),
            ("user.B204-Text-Label-On-Image-Dir", "classification", False),
        )
        for name, label, text_dir in cases:
            dest = os.path.join(_opts.hf_cache, "models--" + name[len("user.") :])
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            if text_dir:
                _write_stub_text_model_dir(dest)
            else:
                _write_stub_model_dir(dest)
            response = requests.post(
                f"{self.base_url}/pull",
                json={
                    "model_name": name,
                    "recipe": RECIPE,
                    "labels": [label],
                    "local_import": True,
                },
                timeout=TIMEOUT_DEFAULT,
            )
            self.assertEqual(response.status_code, 200, response.text[:500])
            response = requests.post(
                f"{self.base_url}/load",
                json={"model_name": name},
                timeout=TIMEOUT_DEFAULT,
            )
            self.assertGreaterEqual(response.status_code, 400, response.text[:500])
            wanted = "image-classifier" if text_dir else "text-classifier"
            found = "text-classifier" if text_dir else "image-classifier"
            self.assertIn(f"needs the {wanted} layout", response.text)
            self.assertIn(f"has the {found} layout", response.text)
            print(f"[OK] {label} on a {found} dir refused: {response.status_code}")

    def _register_stub(self, name, label, writer):
        dest = os.path.join(_opts.hf_cache, "models--" + name[len("user.") :])
        if os.path.lexists(dest):
            shutil.rmtree(dest)
        writer(dest)
        return requests.post(
            f"{self.base_url}/pull",
            json={
                "model_name": name,
                "recipe": RECIPE,
                "labels": [label],
                "local_import": True,
            },
            timeout=TIMEOUT_DEFAULT,
        )

    def test_517_engine_without_image_route_is_a_502(self):
        # An engine release older than its first image release answers 404 on
        # /classify/image; the client must learn which release it needs.
        if not _opts.fake_backend or not _opts.hf_cache:
            self.skipTest("needs --fake-backend and --hf-cache")
        name = "user.B204-Old-Engine-Image"

        def writer(dest):
            _write_stub_model_dir(dest)
            with open(os.path.join(dest, "FAKE_NO_IMAGE_ROUTE"), "w", encoding="utf-8"):
                pass

        response = self._register_stub(name, "image-classification", writer)
        self.assertEqual(response.status_code, 200, response.text[:500])
        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"model": name},
        )
        self._assert_status(response, 502, "engine without /classify/image")
        error = response.json()["error"]
        self.assertEqual(error["code"], "image_classification_unsupported")
        self.assertIn(ENGINE["image_release"], error["message"])

    def test_518_invalid_image_declarations_never_load(self):
        if not _opts.hf_cache:
            self.skipTest("no --hf-cache (or LEMONADE_TEST_HF_CACHE)")
        if os.name == "nt":
            self.skipTest("needs symlinks")
        outside = os.path.abspath(
            os.path.join(_opts.hf_cache, "..", f"b204-outside-{RECIPE}")
        )
        os.makedirs(outside, exist_ok=True)
        with open(os.path.join(outside, "labels.txt"), "w", encoding="utf-8") as f:
            f.write("background\nmilitary uniform\n")
        with open(os.path.join(outside, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"task": "image-classification", "labels_file": "labels.txt"}, f)

        def bad_labels(dest):
            _write_stub_text_model_dir(dest)
            with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"task": "image-classification", "labels_file": "../outside.txt"},
                    f,
                )

        def linked_manifest(dest):
            _write_stub_model_dir(dest)
            os.remove(os.path.join(dest, "manifest.json"))
            os.symlink(
                os.path.join(outside, "manifest.json"),
                os.path.join(dest, "manifest.json"),
            )

        cases = (
            ("user.B204-Bad-Labels-Text-Files", "classification", bad_labels),
            ("user.B204-Bad-Labels-Text-Files-Img", "image-classification", bad_labels),
            ("user.B204-Linked-Manifest", "image-classification", linked_manifest),
        )
        for name, label, writer in cases:
            response = self._register_stub(name, label, writer)
            if response.status_code == 200:
                response = requests.post(
                    f"{self.base_url}/load",
                    json={"model_name": name},
                    timeout=TIMEOUT_DEFAULT,
                )
            self.assertGreaterEqual(
                response.status_code, 400, f"{name}: {response.text[:500]}"
            )
            print(f"[OK] {name} ({label}) refused: {response.status_code}")

    def test_513_repeated_text_crashes_pause_the_model(self):
        self._require_model()
        self._require_text_model()
        crashes = os.path.join(_opts.fake_state_dir, "crashes.log")
        starts = os.path.join(_opts.fake_state_dir, "starts.log")
        response = None
        for _ in range(4):
            response = self._classify_text(POISON_TEXT)
            if response.status_code != 502:
                break
        self._assert_status(response, 503, "poison text after repeated crashes")
        self.assertEqual(response.json()["error"]["code"], "backend_crash_cooldown")
        self.assertTrue(1 <= int(response.headers["Retry-After"]) <= 60)

        crashes_before, starts_before = _line_count(crashes), _line_count(starts)
        self._assert_status(self._classify_text("hello"), 503, "good text while paused")

        # A registered router that uses the paused model as its classifier takes
        # its on_error path; it must not restart the model to find that out.
        # Routing loads a helper only for a registered policy, hence /pull.
        collection = "user.B194-Text-Router"
        policy = {
            "version": "1",
            "model_name": collection,
            "recipe": "collection.router",
            "components": [TEXT_MODEL, MODEL],
            "routing": {
                "candidates": [MODEL],
                "default_model": MODEL,
                "classifiers": [
                    {
                        "id": "safety",
                        "type": "classifier",
                        "model": TEXT_MODEL,
                        "labels": ["safe", "unsafe"],
                        "default_label": "unsafe",
                        "on_error": "match_false",
                    }
                ],
                "rules": [
                    {
                        "id": "unsafe",
                        "match": {"classifier": "safety", "min_score": 0.5},
                        "route_to": MODEL,
                    }
                ],
            },
        }
        # Offline, /pull registers the definition and answers lemond_offline.
        response = requests.post(f"{self.base_url}/pull", json=policy, timeout=600)
        if response.status_code != 200:
            self._assert_status(response, 400, "register the router collection")
            self.assertEqual(response.json().get("code"), "lemond_offline")
        try:
            response = requests.post(
                f"{self.base_url}/routing/validate",
                json={"model": collection, "prompt": "hello"},
                timeout=TIMEOUT_CLASSIFY,
            )
            self._assert_status(response, 200, "routing/validate while paused")
        finally:
            requests.post(
                f"{self.base_url}/delete",
                json={"model_name": collection},
                timeout=TIMEOUT_DEFAULT,
            )

        self.assertEqual(
            _line_count(crashes), crashes_before, "paused model crashed again"
        )
        self.assertEqual(
            _line_count(starts), starts_before, "paused model was reloaded"
        )

    def test_514_large_multipart_image_is_accepted(self):
        # 12 MiB is under the 16 MiB image cap and the 23 MiB body cap.
        self._require_model()
        image = _jpeg_like(b"\x01" * (12 * 1024 * 1024))
        response = self._post_form(
            {"image": ("big.jpg", image, "image/jpeg")},
            data={"model": MODEL, "top_k": "1"},
        )
        if _opts.fake_backend:
            self._assert_status(response, 200, "12 MiB multipart image")
            return
        # A real decoder refuses the synthetic JPEG; its 400 shows the body
        # passed lemond's cap and reached the engine.
        self._assert_status(response, 400, "12 MiB multipart image")
        self.assertIn("JPEG", response.json()["error"]["message"])


if __name__ == "__main__":
    _opts = _parse_own_args()
    RECIPE = _opts.wrapped_server
    ENGINE = ENGINES[RECIPE]
    MODEL = ENGINE["model"]
    TEXT_MODEL = ENGINE["text_model"]
    run_server_tests(
        ImageClassifyTests,
        "IMAGE CLASSIFICATION TESTS",
        modality="image_classification",
        default_wrapped_server="tflite",
    )
