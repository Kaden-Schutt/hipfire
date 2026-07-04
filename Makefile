.PHONY: build install dev-install dev-unlink

HIPFIRE_DIR ?= $(HOME)/.hipfire
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TARGET_DIR ?= $(REPO_ROOT)/target/release
# Binaries the installer places in ~/.hipfire/bin (and that `hipfire serve`
# resolves the daemon from). Override DEV_BINS to link a different set.
DEV_BINS ?= hipfire hipfire-daemon hipfire-eval hipfire-tui hipfire-system-monitor hipfire-host-profile

build:
	cargo build --release

install:
	./install.sh

# Dev install: symlink each ~/.hipfire/bin/<bin> to the freshly built binary in
# the source tree's target/release. After this runs once, a plain
# `cargo build --release` (or `make build`) updates the installed binaries in
# place — no reinstall — so a running `hipfire serve` picks up daemon changes
# (new quant_type support, kernels, etc.) on its next daemon spawn. Far faster
# than `make install`, which does a from-scratch `cargo install`.
dev-install: build
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
