# qw35 build workflow — release-only, macOS 14 (Apple-silicon M1+) floor.
#
#   make release   build the optimized server, then wipe target/debug (~670 MB)
#                  while keeping the warm release cache so rebuilds stay ~fast.
#   make clean-debug   drop only the debug artifacts.
#   make clean         full wipe (next build recompiles ALL deps from scratch).
#   make run           build release + run the server.
#   make dmg           full distributable: qw35.app (server + agent + first-run
#                      model-download GUI, see packaging/) inside an ad-hoc
#                      signed DMG at dist/qw35-<version>-arm64.dmg.
#   make app / icns / packaging-venv   the dmg pipeline's individual stages.
#
# MACOSX_DEPLOYMENT_TARGET pins the binary's min-OS to the same floor build.rs
# stamps into the embedded metallib, so the release loads on macOS 14+.

export MACOSX_DEPLOYMENT_TARGET := 14.0

CARGO      ?= cargo
TARGET_DIR := target
PKG_VENV   := packaging/.venv

.PHONY: release clean-debug clean run packaging-venv icns app dmg

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

$(PKG_VENV)/bin/pyinstaller:
	python3 -m venv $(PKG_VENV)
	$(PKG_VENV)/bin/pip install -q -r packaging/requirements-app.txt

packaging-venv: $(PKG_VENV)/bin/pyinstaller

packaging/qw35.icns: assets/app_icon.png packaging/make_icns.py | packaging-venv
	$(PKG_VENV)/bin/python packaging/make_icns.py assets/app_icon.png packaging/qw35.icns

icns: packaging/qw35.icns

app: release packaging-venv icns
	cd packaging && .venv/bin/pyinstaller --noconfirm --distpath ../dist --workpath ../$(TARGET_DIR)/pyinstaller qw35.spec
	codesign --force -s - dist/qw35.app/Contents/Frameworks/bin/qw35 2>/dev/null || \
		codesign --force -s - dist/qw35.app/Contents/Resources/bin/qw35
	codesign --force --deep -s - dist/qw35.app
	@echo "qw35: app bundle -> dist/qw35.app ($$(du -sh dist/qw35.app | cut -f1))"

dmg: app
	sh packaging/make_dmg.sh dist/qw35.app dist
