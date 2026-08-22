# Quantizer image: gate-runner base + patched source + prebuilt binary.
FROM hipfire-gate

SHELL ["/bin/bash", "-c"]
# Overlay the CURRENT checkout (the base image froze an older snapshot),
# then build so the fix is actually compiled in.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN source /root/.cargo/env && cd /hipfire && \
    touch crates/hipfire-quantize/src/pipeline.rs && \
    cargo build --release --locked -p hipfire-quantize && \
    cp target/release/hipfire-quantize /usr/local/bin/

ENTRYPOINT ["/usr/local/bin/hipfire-quantize"]
