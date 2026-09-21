# HiveBear — CPU-only Docker image
# Usage: docker run -it --rm -p 11434:11434 ghcr.io/beckhamlabsllc/hivebear quickstart

# --- Builder stage ---
FROM rust:1.97-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release -p hivebear-cli \
    && strip target/release/hivebear

# --- Runtime stage ---
FROM debian:bookworm-slim

# libgomp1: llama.cpp is built with OpenMP, so the binary links libgomp.so.1.
# The builder stage gets it via g++; this stage does not, and without it the
# image builds perfectly and then dies on every run with
# "error while loading shared libraries: libgomp.so.1".
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/hivebear /usr/local/bin/hivebear

# Create the non-root user and its data directory BEFORE declaring the volume.
# VOLUME does not create the directory, so the chown had nothing to act on
# ("chown: cannot access '/data'"), and anything written to a volume path after
# VOLUME is declared is discarded by the builder anyway.
RUN groupadd -r hivebear && useradd -r -g hivebear -d /data -s /sbin/nologin hivebear \
    && mkdir -p /data \
    && chown -R hivebear:hivebear /data

# Default model storage inside container
ENV HIVEBEAR_DATA_DIR=/data
VOLUME /data

USER hivebear

EXPOSE 11434

ENTRYPOINT ["hivebear"]
CMD ["quickstart"]
