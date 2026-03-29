# HiveBear — CPU-only Docker image
# Usage: docker run -it --rm -p 11434:11434 ghcr.io/beckhamlabsllc/hivebear quickstart

# --- Builder stage ---
FROM rust:1.83-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release -p hivebear-cli \
    && strip target/release/hivebear

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/hivebear /usr/local/bin/hivebear

# Default model storage inside container
ENV HIVEBEAR_DATA_DIR=/data
VOLUME /data

# Create non-root user for security
RUN groupadd -r hivebear && useradd -r -g hivebear -d /data -s /sbin/nologin hivebear \
    && chown -R hivebear:hivebear /data

USER hivebear

EXPOSE 11434

ENTRYPOINT ["hivebear"]
CMD ["quickstart"]
