"""
Image-classification tests for Lemonade Server: POST /v1/images/classify with
the tflite recipe (tflite-server >= 0.2.0 subprocess).

Usage:
    python server_image_classify.py --wrapped-server tflite --backend system \\
        --image-model-dir <dir> --hf-cache <lemond's HF cache dir>

    <dir> is a tflite-server image model directory (model.tflite, labels.txt,
    manifest.json), such as the one tflite-server's tools/make_image_model_dir.py
    writes for TF MobileNetV2 1.0 224. It is copied into the server's HF cache
    and registered with a local import as user.MobileNetV2-1.0-224-TFLite, so the
    server must run on this machine. LEMONADE_TEST_IMAGE_MODEL_DIR and
    LEMONADE_TEST_HF_CACHE work in place of the flags.

    --fake-backend runs the same suite against test/utils/fake_tflite_server.py
    installed on the server's PATH as `tflite-server`. It skips the accuracy
    assertions and enables the crash non-replay test, which needs a backend that
    crashes on demand; pass --fake-state-dir, the FAKE_TFLITE_STATE_DIR the server
    was started with.

    --lemond-pid enables the resident-memory assertion on the oversized upload.

Negative tests need no model and run first. Positive tests skip, with the
reason, when no model directory or no tflite-server is available.

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
import socket
import struct
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

MODEL = "user.MobileNetV2-1.0-224-TFLite"
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_image_grace_hopper.jpg"
)
REFERENCE_INDEX = 653
REFERENCE_SCORE = 0.803491
REFERENCE_TOL = 0.02
POISON = b"B169-POISON-IMAGE"

_opts = argparse.Namespace(
    image_model_dir=None,
    hf_cache=None,
    fake_backend=False,
    fake_state_dir=None,
    lemond_pid=None,
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
        sock.sendall(headers.encode() + body)
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
    with open(os.path.join(path, "model.tflite"), "wb") as f:
        f.write(b"stub")
    with open(os.path.join(path, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("background\nmilitary uniform\n")
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"task": "image-classification", "labels_file": "labels.txt"}, f)
    return os.path.abspath(path)


def _vm_rss_kb(pid):
    with open(f"/proc/{pid}/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


class ImageClassifyTests(ServerTestBase):
    """Tests for the /images/classify endpoint."""

    _registered = False

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
                os.path.join(_opts.hf_cache, "..", "b169-stub-image-model")
            )
        if not _opts.image_model_dir or not os.path.isdir(_opts.image_model_dir):
            self.skipTest("no --image-model-dir (or LEMONADE_TEST_IMAGE_MODEL_DIR)")
        if not _opts.hf_cache:
            self.skipTest("no --hf-cache (or LEMONADE_TEST_HF_CACHE)")
        if shutil.which("tflite-server") is None:
            self.skipTest("tflite-server is not on PATH")
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
                "recipe": "tflite",
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
        load_model(MODEL, tflite_args="--max-image-pixels 1000")
        try:
            response = self._post_form(
                {"image": ("p.png", _png(64, 64), "image/png")},
                data={"model": MODEL},
            )
            self._assert_status(response, 400, "64x64 PNG over a 1000-pixel cap")
        finally:
            unload_model(MODEL)
            load_model(MODEL, tflite_args="")

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

        response = self._post_form(
            {"image": ("g.jpg", self._fixture(), "image/jpeg")},
            data={"model": MODEL, "top_k": "1"},
        )
        self._assert_reference(response, 1)
        self.assertEqual(
            lines(starts) - starts_before, 1, "expected exactly one reload"
        )


if __name__ == "__main__":
    _opts = _parse_own_args()
    run_server_tests(
        ImageClassifyTests,
        "IMAGE CLASSIFICATION TESTS",
        modality="image_classification",
        default_wrapped_server="tflite",
    )
