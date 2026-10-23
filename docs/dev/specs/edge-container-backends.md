# Edge Container Backends Spec

- [Summary](#summary)
- [Proposal Size](#proposal-size)
- [User Story](#user-story)
- [Scope](#scope)
- [Backends](#backends)
- [Changes to the Container Backends Spec](#changes-to-the-container-backends-spec)
- [Engine Contract](#engine-contract)
- [Decision and Zero-Shot Models](#decision-and-zero-shot-models)
- [NPU Devices](#npu-devices)
- [Gateway Profile](#gateway-profile)
- [prpl Life Cycle Management](#prpl-life-cycle-management)
- [Breaking Changes](#breaking-changes)
- [Maintenance Plan](#maintenance-plan)
- [Risks](#risks)
- [Open Questions](#open-questions)

## Summary

This spec proposes three CPU classification and decision backends for gateways and other small Linux hosts, delivered in the container backend format of the [Container Backends Spec](./container-backends.md) (RFC #3585).

- The backends are `tflite:edge`, `executorch:edge` and `onnxruntime:edge`, each a `guest` tier, `container` format backend.
- Each backend runs from a digest-pinned, multi-arch (amd64 and arm64) OCI image, through the `ContainerManager` that #3709 adds.
- The first deployment target is prplOS, the prpl Foundation's OpenWrt-based gateway OS, whose Life Cycle Management (LCM) already runs OCI containers. Netstatz wires `lemond` into prplOS on its own fork first; this spec covers only what upstream Lemonade would carry.

## Proposal Size

Major feature, delivered as one PR per backend after the container backend stack (#3709 and its follow-ups) merges.

## User Story

A gateway vendor ships `lemond` in the gateway's base image and gives on-device applications one local inference API for small classifiers and decision models, such as phishing text detection, image tagging, and choosing the cause of a network fault from a short list. The engines arrive as container images, so the vendor installs, updates and removes them through the same tooling it uses for every other containerized application. A request for a model the gateway cannot hold goes to a LAN `lemond` or a cloud provider through the existing `cloud_providers` setting.

Release note: "Three guest CPU backends for text, image and zero-shot classification and for decision models (`tflite:edge`, `executorch:edge`, `onnxruntime:edge`), shipped as amd64 and arm64 container images, for small Linux hosts such as home gateways."

## Scope

This section states what the beta covers.

| Item | In the beta |
| --- | --- |
| Platform | Linux amd64 and arm64, CPU only |
| Container tools | Podman and Docker, exactly as the [Container Backends Spec](./container-backends.md) defines them |
| Modes | Text classification and zero-shot classification (`POST /v1/classify`), image classification (`POST /v1/images/classify`, `tflite:edge` only), decision questions (`POST /v1/systemone`) |
| Models | Downloaded by Lemonade and mounted read-only, per the Container Backends Spec |
| Image source | One new namespace, `ghcr.io/ianbmacdonald/*`, until the maintainers choose a home (see [Open Questions](#open-questions)) |
| prpl LCM as a container tool | Not in the beta (see [prpl Life Cycle Management](#prpl-life-cycle-management)) |

## Backends

Each backend is a new `support` row with its own descriptor labels. `tflite` and `executorch` are new recipes; `onnxruntime` is the existing recipe with a new backend beside its native `cpu` backend.

| Backend | Tier | Format | Image | Engine | Modes |
| --- | --- | --- | --- | --- | --- |
| `tflite:edge` | guest | container | `ghcr.io/ianbmacdonald/tflite-server` | LiteRT, CPU with XNNPACK | `classification`, `image-classification`, `systemone` |
| `executorch:edge` | guest | container | `ghcr.io/ianbmacdonald/et-server` | ExecuTorch, CPU with XNNPACK | `classification`, `systemone` |
| `onnxruntime:edge` | guest | container | `ghcr.io/ianbmacdonald/ort-server` | ONNX Runtime CPU | `classification`, `systemone`, for `model.onnx` and `model.ort` files |

Each backend's `ContainerPolicy` is:

| Field | Value |
| --- | --- |
| `image` | The image in the table above |
| `devices` | `{}` |
| `cap_add` | `{}` |

Each image is `FROM scratch` with the engine's musl build and its shared libraries, built from the engine's own repository by that repository's CI, for `linux/amd64` and `linux/arm64`, and pinned in `backend_versions.json` as `<tag>@<digest>`.

## Changes to the Container Backends Spec

This section lists the edits the [Container Backends Spec](./container-backends.md) needs to admit these backends. Each is a widening, and none changes a GPU container backend.

| Spec section | Today | Proposed |
| --- | --- | --- |
| Scope, Platform | Linux-only, GPU-only, single GPU | Linux-only; GPU or CPU; single device |
| `containers` Descriptor Field, `devices` | Required, at least one device node | Required; `{}` declares a CPU-only container, which gets no `--device` and no `--group-add` |
| `containers` Descriptor Field, `image` | Only `docker.io/kyuz0/*` and `ghcr.io/peonist-ai/*` | Adds `ghcr.io/ianbmacdonald/*` |
| Run Options, `--env HIP_VISIBLE_DEVICES` | Set when `devices` includes `/dev/kfd` | Unchanged; never set for a CPU-only container |

## Engine Contract

This section defines what the three engine images share, so one `WrappedServer` pattern serves all three. All three engines follow the `lemonade-sdk/ort-server` contract; `tflite-server` and `et-server` are Apache-2.0 derivatives of `ort-server` and credit it in their `NOTICE` files.

| Property | Value |
| --- | --- |
| Program | `<engine> --model-path /mnt/models/<model> --port <port> --host 0.0.0.0` |
| Ready endpoint | `GET /health`, which returns `{"status": "ok", "task": "<task>"}` |
| Text classification | `POST /classify` |
| Zero-shot classification | `POST /classify` with a `labels` array, the field #3952 has `lemond` send to `ort-server` |
| Decision questions | `POST /systemone`, with the request and response of llama.cpp's `POST /v1/systemone`: a `state` and named `questions` of type `choice`, `score` or `noul`, answered with one entry per question in `answers` |
| Image classification | `POST /classify/image`, multipart (`tflite-server` only) |
| Model directory | Manifest, tokenizer files and one model file (`model.tflite`, `model.pte`, `model.onnx` or `model.ort`), mounted one file at a time by the Container Backends Spec's directory rule |
| Writable state | `tflite-server`'s XNNPACK weight cache goes to `/tmp` inside the container, never beside the read-only model |

These engine changes are required before the images ship, one release per engine:

1. Add a `--host` flag. Today each engine binds `127.0.0.1`, which a container's published port cannot reach.
2. In `tflite-server`, add a weight cache path flag. Today the cache is written beside the model.
3. Accept `labels` on `POST /classify`. `onnxruntime:edge` takes this from `ort-server` once #3952 lands; `tflite-server` and `et-server` port it.
4. Add `POST /systemone` to all three engines.
5. Build each engine with its framework's full default operator set. The curated operator sets on the Netstatz fork refuse zero-shot models such as GLiClass.

## Decision and Zero-Shot Models

This section describes the two model types that need the new engine endpoints, and the `lemond` routes they reach through.

A decision model scores options the caller supplies instead of generating text. ggml-org's decision models (Julia-1, 144M, mmBERT-small; Laya, 421M, ModernBERT-large) are encoders of DistilBERT's class, so an export to ONNX, LiteRT or ExecuTorch that keeps the decision head should run on these CPU engines. No such export has run yet. A zero-shot classifier scores the input against labels the caller declares per request.

The target models are:

| Model | Type | Backends | Status |
| --- | --- | --- | --- |
| Julia-1 (144M) | Decision | All three | Export with the decision head not yet done |
| Laya (421M) | Decision | All three | Export with the decision head not yet done |
| `knowledgator/gliclass-modern-base-v3.0` (151M) | Zero-shot | `onnxruntime:edge` | Export not yet done; needs the full operator set |

The `lemond` routes come from two open PRs, and each needs one widening:

| PR | Adds | Widening this spec needs |
| --- | --- | --- |
| #3952 | `labels` on `POST /v1/classify`, and a `zero_shot` router classifier | None for `onnxruntime:edge`. `tflite:edge` and `executorch:edge` accept `labels` once their engines do |
| #3956 | `POST /v1/systemone`, answered by a `llamacpp` model labelled `systemone` | Also route `/v1/systemone` to any backend whose descriptor lists the `systemone` mode |

## NPU Devices

This section describes how an NPU fits these backends when gateway NPUs arrive. The beta is CPU only and adds nothing for NPUs.

An NPU backend is a separate `support` row per NPU vendor, beside the CPU row, such as `tflite:edge-<vendor>`:

| Item | NPU row |
| --- | --- |
| Device type | `DEVICE_NPU`, with a new device key per vendor beside today's `amd_npu` |
| `ContainerPolicy.devices` | The vendor's NPU device nodes |
| Image | The CPU image plus the vendor's delegate or execution provider libraries |
| Detection | `lemond` detects the vendor's NPU before it offers the row, as it does for `amd_npu` |
| Scheduling | The Router's existing NPU exclusivity applies |

The engines' delegates (LiteRT), backends (ExecuTorch) and execution providers (ONNX Runtime) already select the accelerator at load, so an NPU row changes the image and the device nodes, not the engine contract.

## Gateway Profile

This section lists the configuration changes a gateway needs. Each is a separate PR or RFC, and none is required by the beta.

| Change | Kind | Status on the Netstatz fork |
| --- | --- | --- |
| `offline` also blocks the implicit model download on `/load` and first request | Bug fix | Fixed, with a cpp-ci test |
| Storage policy: model downloads `none`, `budget` or `allowlist`, with routing kept available when downloads are off | Separate RFC; changes config semantics | Designed, not built |
| Request body limit, request receive deadline and per-client connection cap for a LAN-facing `lemond` | Separate RFC; new config keys | Built and released on the fork |
| `/pull` pinned to an exact Hugging Face revision | Small feature | Built and released on the fork |

## prpl Life Cycle Management

This section explains how the container format maps onto prplOS. It is context for the reviewers; the beta adds nothing for it.

prplOS installs and runs containers through LCM. An install request (`SoftwareModules.InstallDU` in the TR-181 data model) pulls an OCI image (the `Rlyeh` component), and `Cthulhu` runs it as an LXC container on an LCM bridge. Netstatz has deployed an application container this way on a prplWare v5 x86_64 gateway.

| Container Backends Spec concept | prplOS LCM counterpart |
| --- | --- |
| Install pulls the pinned image; a load never downloads | The platform installs the deployment unit; `lemond` never installs on prplOS |
| `ContainerManager` runs `podman run` or `docker run` | A container tool that starts and stops an installed unit through the LCM data model |
| `--env` for engine settings | `InstallDU` `EnvVariable` |
| Private network per container | An LCM bridge |
| Models mounted read-only | Not yet mapped |

A prpl LCM container tool in `ContainerManager` is a follow-up RFC, written once Netstatz has run it on its fork. The beta's images are the same images prplOS would install, so the engine work is shared.

Each image is `FROM scratch` with the engine's musl build, so one build per architecture serves three paths: the native package prplOS installs today, a container under Podman or Docker on any Linux host, and a container under LCM. `lemond` itself stays native on prplOS; its aarch64 musl build is released on the Netstatz fork. Not yet verified: a `scratch` musl engine image running under LCM on an aarch64 prplOS gateway.

## Breaking Changes

None. The changes widen the Container Backends Spec and add backends; no existing backend's behaviour changes.

## Maintenance Plan

This section maps the backends onto the backend policy proposed in [#3925](https://github.com/lemonade-sdk/lemonade/pull/3925) (support tiers, test plan, maintenance plan).

| Item | Plan |
| --- | --- |
| Tier | `guest`: maintained by Netstatz, without a support guarantee from Lemonade's maintainers |
| Sandbox | The `container` format meets the guest tier's sandbox requirement |
| Test plan | A smoketest per backend on GitHub-hosted runners: install, load, classify, unload. CPU only, so no self-hosted or DevLab Dispatch hardware is needed |
| Engine updates | Each engine's CI publishes a multi-arch image per release; a PR updates the pin in `backend_versions.json` |
| New models | Registry entries only; the engines need no per-model code |
| Promote to `experimental` | A Lemonade maintainer agrees to own the backend, for example after prplOS adopts it, or Android or ChromeOS targets ask for the same engines |
| Remove | No adopter by a date the maintainers set when merging |

## Risks

| Risk | Mitigation |
| --- | --- |
| A LAN-facing gateway `lemond` | The gateway limits RFC in [Gateway Profile](#gateway-profile) |
| Images from a namespace outside the existing allowlist | Digest pins, and the run options the Container Backends Spec applies to every container (`--cap-drop=all`, `no-new-privileges`, private network) |
| Image size on a gateway's flash | Measured per image before the first release; the beta targets hosts with a data volume |
| Maintenance if no adopter appears | The removal rule in [Maintenance Plan](#maintenance-plan) |

## Open Questions

These are the decisions that belong to the maintainers.

1. **Image home.** Should the three images publish under `lemonade-sdk` beside `ort-server`, or under `ghcr.io/ianbmacdonald` as guest images?
2. **Native fallback.** The engines also build as native binaries. Should a `native` format row for each backend sit beside the container row, for hosts without Podman or Docker?
3. **Beta gate.** Should the backends wait for the full container backend stack to merge, or land behind `enable_containers` as soon as #3709 merges?
