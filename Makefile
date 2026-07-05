.PHONY: all install link dev-unlink ui serve daemon quant eval tui monitor help

HIPFIRE_DIR ?= $(HOME)/.hipfire
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TARGET_DIR ?= $(REPO_ROOT)/target/release
# Binaries symlinked into ~/.hipfire/bin (and that `hipfire serve` resolves the
# daemon from). Override DEV_BINS to link a different set.
DEV_BINS ?= hipfire hipfire-daemon hipfire-quantize hipfire-eval hipfire-tui hipfire-system-monitor hipfire-host-profile

# Embedded browser UIs (Leptos/WASM) baked into the `hipfire` CLI. They require
# `trunk` + the wasm32-unknown-unknown target; when trunk is absent we build
# without the embedded UIs (the server falls back to its lightweight routes).
TRUNK := $(shell command -v trunk 2>/dev/null)
ifeq ($(TRUNK),)
  CLI_FEATURES :=
else
  CLI_FEATURES := chat-ui-embed,admin-ui-embed
endif
CLI_FEATURE_ARG := $(if $(CLI_FEATURES),--features $(CLI_FEATURES),)

# ── Default: build EVERYTHING with all features (incl. the embedded browser
#    UIs) and refresh the dev symlinks in ~/.hipfire/bin so a running
#    `hipfire serve` picks up the rebuilt binaries on its next daemon spawn.
#    Plain `make` updates the whole dev install in place.
all: ui
	cargo build --release
	cargo build --release -p hipfire-cli --bin hipfire $(CLI_FEATURE_ARG)
	@$(MAKE) --no-print-directory link

# Full from-scratch install into ~/.hipfire (cargo install, all features incl.
# UIs). Use this for a clean/first install; `make` (dev symlinks) is faster for
# iterating.
install:
	./install.sh

# Build the Leptos/WASM browser UIs into their dist/ dirs (embedded via
# CLI_FEATURES). No-op with a note when trunk is unavailable.
ui:
ifeq ($(TRUNK),)
	@echo "trunk not found — building without embedded browser UIs."
	@echo "  enable: rustup target add wasm32-unknown-unknown && cargo install trunk"
else
	cd "$(REPO_ROOT)/crates/hipfire-admin-ui" && env -u NO_COLOR trunk build --release
	cd "$(REPO_ROOT)/crates/hipfire-chat-ui"  && env -u NO_COLOR trunk build --release
endif

# ── Per-component fast rebuilds. The dev symlinks point at target/release, so a
#    rebuild updates the installed binary in place. Run plain `make` once first
#    to create the symlinks (each target re-links defensively anyway).

# Rebuild the serving path: hipfire (with UIs) + daemon.
serve: ui
	cargo build --release -p hipfire-daemon
	cargo build --release -p hipfire-cli --bin hipfire $(CLI_FEATURE_ARG)
	@$(MAKE) --no-print-directory link

daemon:
	cargo build --release -p hipfire-daemon
	@$(MAKE) --no-print-directory link

quant:
	cargo build --release -p hipfire-quantize
	@$(MAKE) --no-print-directory link

eval:
	cargo build --release -p hipfire-eval
	@$(MAKE) --no-print-directory link

tui:
	cargo build --release -p hipfire-tui
	@$(MAKE) --no-print-directory link

monitor:
	cargo build --release -p hipfire-system-monitor
	@$(MAKE) --no-print-directory link

# ── dev symlink management ──────────────────────────────────────────────
# Symlink each ~/.hipfire/bin/<bin> to the freshly built binary in
# target/release. Idempotent; safe to run repeatedly.
link:
	@mkdir -p "$(HIPFIRE_DIR)/bin"
	@for b in $(DEV_BINS); do \
		if [ -e "$(TARGET_DIR)/$$b" ]; then \
			ln -sfn "$(TARGET_DIR)/$$b" "$(HIPFIRE_DIR)/bin/$$b"; \
			echo "linked $(HIPFIRE_DIR)/bin/$$b -> $(TARGET_DIR)/$$b"; \
		else \
			echo "skip $$b (no $(TARGET_DIR)/$$b)"; \
		fi; \
	done

# Remove the dev symlinks (leaves real installed binaries from `make install`
# untouched; only unlinks entries that are symlinks).
dev-unlink:
	@for b in $(DEV_BINS); do \
		f="$(HIPFIRE_DIR)/bin/$$b"; \
		if [ -L "$$f" ]; then rm -f "$$f"; echo "unlinked $$f"; fi; \
	done

help:
	@echo "make            build everything (all features + embedded UIs) and refresh dev symlinks"
	@echo "make install    full cargo-install into ~/.hipfire (all features)"
	@echo "make serve      rebuild hipfire (with UIs) + hipfire-daemon, relink"
	@echo "make daemon     rebuild hipfire-daemon, relink"
	@echo "make quant      rebuild hipfire-quantize, relink"
	@echo "make eval       rebuild hipfire-eval, relink"
	@echo "make tui        rebuild hipfire-tui, relink"
	@echo "make monitor    rebuild hipfire-system-monitor, relink"
	@echo "make link       (re)create the dev symlinks in ~/.hipfire/bin"
	@echo "make dev-unlink remove the dev symlinks"
