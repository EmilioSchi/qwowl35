#!/bin/sh
# Build the distributable DMG from a built qw35.app.
#
# Usage: sh packaging/make_dmg.sh dist/qw35.app dist
#
# Staging layout: the app, an /Applications symlink for drag-install, and the
# Gatekeeper note (the app is ad-hoc signed, so macOS warns on first open).
set -e

APP=${1:?usage: make_dmg.sh path/to/qw35.app out_dir}
OUT_DIR=${2:?usage: make_dmg.sh path/to/qw35.app out_dir}

VERSION=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")
DMG="$OUT_DIR/qw35-$VERSION-arm64.dmg"

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
cat > "$STAGING/How to open (unsigned app).txt" <<'EOF'
qw35 is not signed with an Apple Developer ID, so macOS will warn the first
time you open it. This is expected; here is how to open it anyway.

First: drag qw35.app onto the Applications folder in this window.

macOS 14 (Sonoma)
  1. In Applications, right-click (or Control-click) qw35.app and choose
     "Open".
  2. In the warning dialog, click "Open".
  This is only needed the first time.

macOS 15 and later (Sequoia, Tahoe)
  1. Double-click qw35.app. macOS will refuse with "Apple could not verify
     ...". Close the dialog (do not move it to the Trash).
  2. Open System Settings -> Privacy & Security, scroll down to the Security
     section, and click "Open Anyway" next to the qw35 message.
  3. Confirm in the next dialog.
  This is only needed the first time.

Terminal alternative (any macOS version):
  xattr -dr com.apple.quarantine /Applications/qw35.app

What the app does on first start: it downloads the two model files it needs
(about 5.4 GB total) from Hugging Face into
~/Library/Application Support/qw35/gguf - it asks first and shows progress.
Requires an Apple-silicon Mac (M1 or newer) with macOS 14+.
EOF

rm -f "$DMG"
hdiutil create -volname "qw35" -srcfolder "$STAGING" -format UDZO -ov "$DMG"
echo "make_dmg: wrote $DMG"
