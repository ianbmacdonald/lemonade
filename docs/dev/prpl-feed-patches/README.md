# lemond prplOS feed patches

`git format-patch` series that turn upstream lemonade into the `feed/prpl-lemond`
tree the prplOS feed builds. Apply both series, in order, with `git am` (or `patch
-p1`) on top of the pinned upstream commit:

| | |
|---|---|
| Base | `lemonade-sdk/lemonade` `main` at `b9a4bc2cd`, "fix(gui3): honor saved ctx_size when loading custom models (#4019) (#4020)" (`b9a4bc2cd210a58fd8b0373476148c95ff088ed2`) |
| Series 1: `3956/` | 1 patch: the net diff of upstream PR lemonade-sdk/lemonade#3956 (SystemOne decision models: `/v1/systemone`, GGUF zero-shot on `/v1/classify`, the `systemone` router classifier) at PR head `9caefc2293beb4d094324dd41a2accb9e64c78b2`, squashed onto the base. #3956 is open, not merged; its own history is full of merges from `main`, so it is carried as one patch, authored by its author. Drop this series once #3956 merges and the base moves past it. |
| Series 2: `fork/` | 58 patches: the ianbmacdonald fork's prpl-demo features, rebased (linearly, merge commits dropped) onto base + series 1. |

```sh
git checkout b9a4bc2cd
git am docs/dev/prpl-feed-patches/3956/*.patch docs/dev/prpl-feed-patches/fork/*.patch
```

The result is byte-identical to `feed/prpl-lemond` minus this directory and
`docs/dev/prpl-feed-dependencies.md` (checked by applying the series to a clean
checkout of the base and diffing). The series builds as a whole; intermediate
patches between 0001 and 0052 do not all compile on their own, because the fork
work was written against the pre-refactor server and is ported to the route-class
server and refactored model manager by later patches (0011-0051 port each feature
as it lands; 0052-0054 finish the backend and build API port).

## musl-only patches

None. Nothing in either series exists only for musl or prplOS: lemond builds for
prplOS 5.1 musl (x86_64 and aarch64) from upstream sources with toolchain and
sysroot configuration alone (see `../prpl-feed-dependencies.md`). Two patches are
generic build fixes that musl just happens to exercise:

- `0053` — `upgradable_http_server.h` called `httplib::detail::shutdown_socket`,
  which cpp-httplib's compiled (split) mode declares only in its implementation
  section; also fixes a duplicate `kBeaconPort`.
- `0057` — lemond took ~6 s to exit on SIGTERM because the eviction-engine and
  VRAM-monitor threads slept out their interval before seeing stop; matters for
  lemond as container PID 1.

## Feature patches (series 2)

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

## Not carried

- The fork's change to the old Electron renderer `src/app/src/renderer/utils/modelLabels.ts`
  (an `image-classification` label): upstream deleted that file in the GUI3 rewrite.
- Generated API reference regions in `docs/api/*.md` for the fork routes are not
  regenerated: `gen_api_boilerplate.py` records live responses and needs real models
  for the new examples. The fork routes' docs live in their `RouteSpec`s.
