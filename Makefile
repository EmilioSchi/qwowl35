# qw35 build workflow — release-only, macOS 14 (Apple-silicon M1+) floor.
#
#   make release   build the optimized server, then wipe target/debug (~670 MB)
#                  while keeping the warm release cache so rebuilds stay ~fast.
#   make clean-debug   drop only the debug artifacts.
#   make clean         full wipe (next build recompiles ALL deps from scratch).
#   make run           build release + run the server.
#
# MACOSX_DEPLOYMENT_TARGET pins the binary's min-OS to the same floor build.rs
# stamps into the embedded metallib, so the release loads on macOS 14+.

export MACOSX_DEPLOYMENT_TARGET := 14.0

CARGO      ?= cargo
TARGET_DIR := target

.PHONY: release clean-debug clean run

release:
	$(CARGO) build --release -p qw35-server
	@rm -rf $(TARGET_DIR)/debug
	@echo "qw35: release binary -> $(TARGET_DIR)/release/qw35 ($$(du -h $(TARGET_DIR)/release/qw35 | cut -f1)); debug build wiped"

clean-debug:
	rm -rf $(TARGET_DIR)/debug

clean:
	$(CARGO) clean

run: release
	$(TARGET_DIR)/release/qw35
