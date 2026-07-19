"""pylsp (python-lsp-server) backend for multilspy — semantic Python depth.

jedi-language-server publishes nothing beyond compile/syntax errors (verified
at the wire: its raw publishDiagnostics payload is ``[]`` even for undefined
names and bad kwargs). pylsp runs pylint + pyflakes, whose severity<=2 rows
carry the semantic findings jedi cannot see — E0602 undefined-variable,
W0102 dangerous-default-value, W0613 unused-argument. multilspy ships no
pylsp backend, but its ``LanguageServer`` base launches any command via
``ProcessLaunchInfo``; this subclass points it at the ``pylsp`` binary.

Two handshake pieces stock multilspy never performs, both REQUIRED — without
either, pylsp publishes empty diagnostic sets:

- answer the server's ``workspace/configuration`` request with the plugin
  settings (multilspy leaves server->client requests unhandled);
- push ``workspace/didChangeConfiguration`` before ``initialized``.

The plugin table is explicit because installed pylsp extensions rewrite the
defaults underneath us: python-lsp-ruff (present in some environments)
auto-disables pyflakes/pycodestyle/mccabe and then, paired with ruff>=0.5,
itself publishes nothing — so ruff is forced off and every linter we rely on
is forced on. Style-only linters stay off: their findings land at severity 2
and would flood the "Warnings (not blocking)" list with whitespace noise.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from multilspy.language_server import LanguageServer
from multilspy.language_servers.jedi_language_server.jedi_server import JediServer
from multilspy.lsp_protocol_handler.server import ProcessLaunchInfo
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger

PYLSP_PLUGINS: dict[str, dict[str, bool]] = {
    "ruff": {"enabled": False},
    "pylint": {"enabled": True},
    "pyflakes": {"enabled": True},
    "pycodestyle": {"enabled": False},
    "flake8": {"enabled": False},
    "pydocstyle": {"enabled": False},
    "mccabe": {"enabled": False},
}


class PylspServer(LanguageServer):
    """multilspy LanguageServer driving the ``pylsp`` binary for Python."""

    def __init__(
        self, config: MultilspyConfig, logger: MultilspyLogger, repository_root_path: str
    ) -> None:
        super().__init__(
            config,
            logger,
            repository_root_path,
            ProcessLaunchInfo(cmd="pylsp", cwd=repository_root_path),
            "python",
        )

    def _get_initialize_params(self, repository_absolute_path: str) -> dict:
        # Reuse jedi's client-capability template (generic LSP client caps,
        # resolved against jedi_server.py's own directory regardless of self).
        return JediServer._get_initialize_params(self, repository_absolute_path)

    @asynccontextmanager
    async def start_server(self) -> AsyncIterator["PylspServer"]:
        async def do_nothing(params):
            return

        async def workspace_configuration(params):
            return [{"plugins": PYLSP_PLUGINS} for _ in params.get("items", [])]

        self.server.on_request("client/registerCapability", do_nothing)
        self.server.on_request("workspace/configuration", workspace_configuration)
        self.server.on_notification("window/logMessage", do_nothing)
        self.server.on_notification("$/progress", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)

        async with super().start_server():
            await self.server.start()
            initialize_params = self._get_initialize_params(self.repository_root_path)
            # No capability asserts here: JediServer's are jedi-specific
            # (completion trigger characters) and pylsp answers differently.
            await self.server.send.initialize(initialize_params)
            self.server.notify.workspace_did_change_configuration(
                {"settings": {"pylsp": {"plugins": PYLSP_PLUGINS}}}
            )
            self.server.notify.initialized({})
            self.completions_available.set()

            yield self

            await self.server.shutdown()
            await self.server.stop()
