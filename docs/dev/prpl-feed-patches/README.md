# lemond prplOS feed patches

`git format-patch` series that turn upstream lemonade into the
`feed/prpl-lemond-no3956` tree the prplOS feed builds. Apply it with `git am --keep-cr`
on top of the pinned upstream commit:

| | |
|---|---|
| Base | `lemonade-sdk/lemonade` `main` at `b9a4bc2cd`, "fix(gui3): honor saved ctx_size when loading custom models (#4019) (#4020)" (`b9a4bc2cd210a58fd8b0373476148c95ff088ed2`) |
| Series: `fork/` | 92 patches: the ianbmacdonald fork's prpl-demo features, rebased (linearly, merge commits dropped) onto the base (0001-0058), then the fixes from the branch's review (0059-0092). |

```sh
git checkout b9a4bc2cd
git am --keep-cr docs/dev/prpl-feed-patches/fork/*.patch
```

`--keep-cr` is needed because 0070 edits `test/cpp/test_routing_helper_reconcile.cpp`,
which has CRLF line endings upstream. `patch -p1` cannot apply the series: 0016 adds a
binary test image (`test/test_image_grace_hopper.jpg`) as a git binary diff.

The result is byte-identical to `feed/prpl-lemond-no3956` minus this directory and
`docs/dev/prpl-feed-dependencies.md` (checked by applying the series to a clean
checkout of the base and diffing). The series builds as a whole; intermediate
patches between 0001 and 0052 do not all compile on their own, because the fork
work was written against the pre-refactor server and is ported to the route-class
server and refactored model manager by later patches (0011-0051 port each feature
as it lands; 0052-0054 finish the backend and build API port).

Upstream PR lemonade-sdk/lemonade#3956 (SystemOne decision models: `/v1/systemone`,
GGUF zero-shot on `/v1/classify`, the `systemone` router classifier) is not carried;
it reaches the feed through upstream `main` once it merges.

## Upgrade notes

- `max_connections_per_client`, `request_receive_timeout` and `max_request_body_mb` must
  be JSON integers in `config.json`. A quoted value such as `"8"` used to read as "off";
  it now stops lemond at startup with
  `<path>/config.json: 'max_connections_per_client' must be an integer (got string)`.
  On OpenWrt/prplOS `config.json` is a conffile, so a hand-edited copy survives the
  upgrade and the failure appears at the first restart; under procd respawn the service
  stays down until the file is fixed. Check existing files for quoted limits before
  upgrading.
- The engine `*_args` (`tflite_args`, `executorch_args`, `litert_args`, `onnxruntime_args`)
  refuse the flags lemond manages, including tflite's and executorch's `--weight-cache`
  and litert's `--cache-dir`. A model whose saved options in `recipe_options.json` carry
  one of them fails to load with a 400 `invalid_backend_args` naming the flag; remove
  the flag from the saved options.

## musl-only patches

None. Nothing in the series exists only for musl or prplOS: lemond builds for
prplOS 5.1 musl (x86_64 and aarch64) from upstream sources with toolchain and
sysroot configuration alone (see `../prpl-feed-dependencies.md`). Two patches are
generic build fixes that musl just happens to exercise:

- `0053` — `upgradable_http_server.h` called `httplib::detail::shutdown_socket`,
  which cpp-httplib's compiled (split) mode declares only in its implementation
  section; also fixes a duplicate `kBeaconPort`.
- `0057` — lemond took ~6 s to exit on SIGTERM because the eviction-engine and
  VRAM-monitor threads slept out their interval before seeing stop; matters for
  lemond as container PID 1.

## Feature patches

| Patches | Feature (fork backlog id) |
|---|---|
| 0001-0006 | LiteRT backend (`litert-lm-server` subprocess) |
| 0007, 0015-0020, 0022, 0025-0027 | TFLite backend (`tflite-server`), `POST /v1/images/classify` (B169) |
| 0008, 0010 | vLLM CDNA (gfx942/gfx950) targets, per-arch release repo override |
| 0009 | ONNX Runtime `model.ort` directories |
| 0011 | per-backend and server memory in `/v1/health` (B168) |
| 0012-0014, 0040-0046 | opt-in LAN beacon listener, listen-list, trusted relays (B170) |
| 0021, 0023-0024 | ExecuTorch backend (`et-server`) (B171) |
| 0028-0034 | public model names in route decisions (B172), classify-only `/routing/validate` for registered routers (B173), CLI pull labels |
| 0035-0039 | verify-once downloads and parallel Range downloads (B132) |
| 0047-0050 | gateway limits: body limit, receive deadline, per-client cap (B179) |
| 0051 | `/pull` `revision` pinning (B174), ported from 22 commits as one |
| 0052-0058 | port to upstream's ServerProcess API, build fixes, regenerated docs, shutdown fix, test allowlists |
| 0059-0076 | fixes from the branch's review rounds |
| 0077 | route spec fixes and the regenerated API reference |
| 0078-0085 | fixes from the branch's fourth review round (`*_args` NUL refusal and 400, classify crash accounting, close-on-exec beacon socket, WebSocket message handling and queue bound, 416 resume recovery, descriptor hygiene in forked backends, the unsecured-bind warning) |
| 0086-0087 | WebSocket warnings logged outside the connection lock, and the watchdog reset kind read before the backend is released |
| 0088 | the config.json path and the JSON type in the refusal of a mistyped gateway limit |
| 0089-0091 | the image forward outlasts the watchdog kill, the watchdog reset kind is published in one step, and a failed ort-server start no longer evicts other models |
| 0092 | a missing ort-server engine no longer evicts other models either |

## Test values

The tests, docs and error examples use made-up private addresses (10.60.0.0/24 and
10.79.0.0/24) and the host name `gw2`, in every patch of the series, not only in the
final tree.

## Not carried

- The fork's change to the old Electron renderer `src/app/src/renderer/utils/modelLabels.ts`
  (an `image-classification` label): upstream deleted that file in the GUI3 rewrite.
