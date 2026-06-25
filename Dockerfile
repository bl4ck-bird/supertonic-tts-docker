# syntax=docker/dockerfile:1
# Build targets (a bare `docker build .` builds the default, minimal):
#   docker build -t supertonic:latest .                  # minimal (wav/pcm)
#   docker build --target ffmpeg -t supertonic:ffmpeg .  # + compressed formats
#
# - helper.rs (inference code) is vendored in src/ (see its header + THIRD_PARTY_LICENSES).
# - onnxruntime is loaded dynamically (ort `load-dynamic`) from Alpine's musl
#   `onnxruntime` package, so the binary is tiny and onnxruntime is NOT compiled.
# - `minimal` (= :latest): wav/pcm only, no ffmpeg.
# - `ffmpeg`: adds ffmpeg for mp3/opus/aac/flac/ogg (opt-in, larger).

# ---- build (musl) ----------------------------------------------------------
FROM rust:1-alpine AS build
# musl-dev: musl libc headers. gcc: ureq's TLS stack uses `ring`, which compiles
# C/asm and needs a C compiler.
RUN apk add --no-cache musl-dev gcc
WORKDIR /app

# Build a DYNAMICALLY-linked musl binary. rust:alpine defaults to a fully static
# binary, but static musl binaries have no dynamic loader, so `dlopen` always
# fails — which breaks `ort`'s load-dynamic (it can't open libonnxruntime).
ENV RUSTFLAGS="-C target-feature=-crt-static"

# 1) Cache dependency compilation against a stub (no onnxruntime needed thanks
#    to load-dynamic, so this is fast). Cargo.lock pins exact versions; --locked
#    fails the build if it is stale, keeping images reproducible.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --locked --bin supertonic \
    && rm -rf src

# 2) Build the real binary. `touch` defeats Docker's mtime/cargo cache trap.
COPY src ./src
RUN find src -name '*.rs' -exec touch {} + \
    && cargo build --release --locked --bin supertonic

# ---- runtime base ----------------------------------------------------------
FROM alpine:edge AS base
ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so.1
# onnxruntime: musl build from edge/community; pulls libstdc++ as a dependency.
# Nothing else is needed at runtime: download + healthcheck are in-process
# (ureq), and its webpki-roots provide CA roots, so no curl and no ca-certificates.
RUN apk add --no-cache onnxruntime

WORKDIR /app
COPY --from=build /app/target/release/supertonic /usr/local/bin/supertonic
COPY webui ./webui

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
  CMD ["supertonic", "healthcheck"]

# The binary reads its SUPERTONIC_* env directly and forwards subcommands: no
# args runs the server, `docker run … synth …` runs the CLI.
ENTRYPOINT ["supertonic"]

# ---- ffmpeg runtime (:ffmpeg) ----------------------------------------------
# Base plus ffmpeg, enabling compressed output formats.
FROM base AS ffmpeg
RUN apk add --no-cache ffmpeg

# ---- minimal runtime (default :latest, wav/pcm) ----------------------------
# Last stage, so a bare `docker build .` produces the minimal image.
FROM base AS minimal
