"""Project-local runtime-scaffold CLI wrapper with the EGRI TraceGuard gate."""

from __future__ import annotations


def main() -> None:
    """Run the upstream runtime-scaffold after installing the RLM TraceGuard integration."""
    from egri.runtime_scaffold_traceguard import install_runtime_scaffold_cli_gate

    install_runtime_scaffold_cli_gate()

    from ouroboros.cli.main import app

    app()
