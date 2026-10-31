# lemond prplOS feed: build dependencies

This lists every external source the `lemond` and `lemonade` build pulls in when it
is cross-compiled for prplOS 5.x (musl, x86_64 and aarch64), as the build resolved
them on branch `feed/prpl-lemond-no3956`, and how to supply each one for an offline
build (the OpenWrt download step fetches, then the build runs with no network).

The build was checked offline: both architectures were configured and built
inside `unshare -rn` (no network namespace) with the sources below supplied through
`FETCHCONTENT_SOURCE_DIR_<NAME>` and `-DFETCHCONTENT_FULLY_DISCONNECTED=ON`.

## Fetched at configure time (CMake FetchContent)

These are fetched only when no suitable system package is found. With the prplOS
sysroot, json, CLI11, zstd, cpp-httplib and mbedTLS are fetched; libcurl and
libwebsockets come from the sysroot.

| Name (`FETCHCONTENT_SOURCE_DIR_<NAME>`) | Repository | Commit | Tag | Licence | Linked |
|---|---|---|---|---|---|
| `JSON` | https://github.com/nlohmann/json | `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` | v3.11.3 | MIT | header-only |
| `CLI11` | https://github.com/CLIUtils/CLI11 | `6c7b07a878ad834957b98d0f9ce1dbe0cb204fc9` | v2.4.2 | BSD-3-Clause | header-only |
| `ZSTD` | https://github.com/facebook/zstd | `f8745da6ff1ad1e7bab384bd1f9d742439278e99` (tag object `ac66b19e6bd6b83238bf008eecc1298105298532` in CMakeLists.txt) | v1.5.7 | BSD-3-Clause (dual GPL-2.0) | static |
| `HTTPLIB` | https://github.com/yhirose/cpp-httplib | `d66d9a95997d51a8ba9822a611d1267757741535` | v0.51.0 | MIT | static (compiled mode) |
| `MBEDTLS` | https://github.com/Mbed-TLS/mbedtls | `107ea89daaefb9867ea9121002fbbdf926780e98` | v3.6.2 | Apache-2.0 (dual GPL-2.0-or-later) | static |
| (mbedTLS submodule `framework`) | https://github.com/Mbed-TLS/mbedtls-framework | `94599c0e3b5036e086446a51a3f79640f70f22f6` | - | Apache-2.0 | build scripts |
| `CURL` (only without system libcurl) | https://github.com/curl/curl | `7161cb17c01dcff1dc5bf89a18437d9d729f1ecd` (tag object `55b5fafb094ebe07ca8a5d4f79813c8b40670795`) | curl-8_5_0 | curl (MIT-style) | not used on prplOS |
| `LIBWEBSOCKETS` (only without system lws) | https://github.com/warmcat/libwebsockets | `4415e84c095857629863804e941b9e1c2e9347ef` | v4.3.3 | MIT | not used on prplOS |
| `BROTLI` (macOS only) | https://github.com/google/brotli | `ed738e842d2fbdf2d6459e39267a633c4a9b2f5d` | v1.1.0 | MIT | not used |

Notes for the feed Makefile:

- The mbedTLS checkout must include its `framework` submodule (`git submodule
  update --init`); a tarball of the bare repository does not build.
- cpp-httplib is built in compiled mode (`HTTPLIB_COMPILE=ON`), which runs its
  `split.py` at configure time, so the build host needs `python3`.
- With the sources in `$DL` laid out as `$DL/json`, `$DL/CLI11`, `$DL/zstd`,
  `$DL/httplib`, `$DL/mbedtls`, the configure line is:

```sh
cmake -G Ninja -S lemonade -B build -DCMAKE_TOOLCHAIN_FILE=prplos-<arch>.cmake \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_WEB_APP=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_JSON=$DL/json \
  -DFETCHCONTENT_SOURCE_DIR_CLI11=$DL/CLI11 \
  -DFETCHCONTENT_SOURCE_DIR_ZSTD=$DL/zstd \
  -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=$DL/httplib \
  -DFETCHCONTENT_SOURCE_DIR_MBEDTLS=$DL/mbedtls
cmake --build build --target lemond lemonade
```

  An OpenWrt feed can instead provide `nlohmann-json`, `cli11`, `zstd` and
  `cpp-httplib` (>= 0.26.0) as host/target packages; CMake then uses them and skips
  the fetch (it reports `Using system ...` for each).

## From the prplOS sysroot (target packages)

| Library | Found by | prplOS package | Runtime `NEEDED` | Licence |
|---|---|---|---|---|
| libcurl >= 8.5.0 (8.12.1 in prplOS 5.1) | pkg-config `libcurl` | `libcurl` | `libcurl.so.4` | curl |
| libwebsockets >= 4.3.3 | pkg-config `libwebsockets` | `libwebsockets-openssl` | `libwebsockets.so.19` | MIT |
| zlib | via libcurl | `zlib` | `libz.so.1` | Zlib |
| libcap | pkg-config `libcap` | `libcap` | `libcap.so.2` | BSD-3-Clause / GPL-2.0 |
| OpenSSL 3 | via libcurl / lws | `libopenssl` | (indirect) | Apache-2.0 |
| libdrm + libdrm_amdgpu | pkg-config `libdrm_amdgpu` | not in the stock image: built static (amdgpu only) from libdrm 2.4.125, MIT | none | MIT |
| libstdc++, libgcc_s, musl | toolchain | `libstdcpp`, `libgcc`, `libc` | `libstdc++.so.6`, `libgcc_s.so.1`, `libc.so` | GPL-3.0 with runtime exception, MIT |

libdrm is only used to read AMD GPU memory; the build requires it to be present.
The feed builds it static with `meson -Damdgpu=enabled` and every other driver
disabled. The sysroot's mbedTLS package is not installed in the image, so its
`.pc` files are hidden from the build to make CMake fetch the static copy above.

## Not fetched at build time

- The web app (`BUILD_WEB_APP=OFF`) and the Tauri app are not built for prplOS, so
  no npm or cargo sources are needed.
- Backend executables (ort-server, tflite-server, et-server, litert-lm-server)
  are separate feed packages. lemond does not download them on prplOS: point
  `<recipe>.cpu_bin` / the `system` backend at the packaged binary.
- `src/cpp/resources/*.json` ship with the build tree and must be installed next to
  `lemond` (`resources/`) or under an install prefix.

## Licence flags

No dependency blocks shipping source in a public feed. mbedTLS and zstd are dual
licensed; the feed can take the Apache-2.0 and BSD-3-Clause terms. Nothing here is
QAIRT or otherwise redistribution-restricted, and the build is CPU-only (libdrm is
read-only GPU memory probing; no GPU runtime is linked).
