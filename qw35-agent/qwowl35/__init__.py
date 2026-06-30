"""
qwowl35 — a minimal terminal coding agent for the qw35-server.

The package is laid out as a flat source root: ``mascot.py`` and the ``tools``
package are imported with bare absolute names (``import mascot``,
``from tools.bash import BashTool``). The launcher in :mod:`qwowl35.__main__`
puts this directory on ``sys.path`` so those imports resolve whether the app is
started with ``python -m qwowl35`` or by running ``__main__.py`` directly.
"""

__version__ = "0.1.0"
