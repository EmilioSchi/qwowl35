# PyInstaller spec for qw35.app (arm64, onedir, windowed).
#
# qwowl35/ ships as DATA, not as traced modules: the package inserts its own
# directory into sys.path and imports its modules by bare name (see
# qwowl35/__main__.py), and its webui/ assets must survive intact. The price
# is that PyInstaller never sees its imports, so every runtime dependency is
# force-collected below.
#
# Build (from packaging/): .venv/bin/pyinstaller --noconfirm qw35.spec

import ast
import importlib.util
import os

from PyInstaller.utils.hooks import collect_all

# SPECPATH is the directory containing this spec file (packaging/).
REPO = os.path.dirname(SPECPATH)

datas, binaries, hiddenimports = [], [], []
for pkg in (
    "textual",
    "rich",
    "textual_serve",  # static/ + templates/ needed at runtime
    "aiohttp",  # textual_serve's HTTP server
    "magika",  # bundles its onnx content-type model
    "onnxruntime",
    "tree_sitter_language_pack",
    "tree_sitter",
    "multilspy",
    "httpx",
    "httpcore",
    "h2",
    "anyio",
    "certifi",
    "xxhash",
    "platformdirs",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h


def _keep(entry):
    src = entry[0]
    return "__pycache__" not in src and f"{os.sep}tests{os.sep}" not in src


qwowl_src = os.path.join(REPO, "qw35-agent", "qwowl35")
datas = [d for d in datas if _keep(d)]


def _qwowl_datas():
    """qwowl35/ as per-file data entries, pruned of caches and tests."""
    out = []
    for dirpath, dirnames, filenames in os.walk(qwowl_src):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "tests")]
        rel = os.path.relpath(dirpath, qwowl_src)
        dest = "qwowl35" if rel == "." else os.path.join("qwowl35", rel)
        for fn in filenames:
            if fn == ".DS_Store" or fn.endswith((".pyc", ".pyo")):
                continue
            out.append((os.path.join(dirpath, fn), dest))
    return out


def _qwowl_hiddenimports():
    """Absolute imports across qwowl35/ sources.

    The package ships as data, so PyInstaller never traces it; without this,
    any stdlib module only qwowl35 uses (html.parser, sqlite3, ...) is absent
    from the frozen app. qwowl35-internal names (imported by bare name via
    its sys.path trick) are filtered out by resolvability: only modules the
    build venv can locate become hidden imports.
    """
    local = {os.path.splitext(n)[0] for n in os.listdir(qwowl_src)}
    names = set()
    for dirpath, dirnames, filenames in os.walk(qwowl_src):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module)
    keep = []
    for name in sorted(names):
        if name.split(".")[0] in local:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                keep.append(name)
        except (ImportError, ValueError, ModuleNotFoundError):
            pass
    return keep


hiddenimports += _qwowl_hiddenimports()
datas += _qwowl_datas()
datas += [(os.path.join(SPECPATH, "launcher", "setup_page"), "setup_page")]
binaries += [(os.path.join(REPO, "target", "release", "qw35"), "bin")]

a = Analysis(
    [os.path.join(SPECPATH, "launcher", "qw35_launcher.py")],
    pathex=[os.path.join(SPECPATH, "launcher")],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "jedi", "jedi_language_server", "pytest", "setuptools"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    name="qw35",
    console=False,
    target_arch="arm64",
    exclude_binaries=True,
)
coll = COLLECT(exe, a.binaries, a.datas, name="qw35")
app = BUNDLE(
    coll,
    name="qw35.app",
    icon=os.path.join(SPECPATH, "qw35.icns"),
    bundle_identifier="com.emilioschi.qw35",
    info_plist={
        "CFBundleName": "qw35",
        "CFBundleDisplayName": "qw35",
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "14.0",
        "LSArchitecturePriority": ["arm64"],
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "© 2026 Emilio Schininà",
    },
)
