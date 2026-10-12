"""Canonical command-line entry point for each service and the local network."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from inference_network.config import CoordinatorSettings, ProviderSettings, Settings
from inference_network.launcher import run_network
from inference_network.observability import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local distributed inference network")
    commands = parser.add_subparsers(dest="command")

    interactive = commands.add_parser(
        "interactive", help="Choose Sender or Provider in the terminal"
    )
    interactive.add_argument(
        "--lan", action="store_true", help="Explicitly enable paired SSH LAN mode"
    )
    interactive.add_argument("--profile", type=Path, help="Private pairing profile file")
    interactive.add_argument(
        "--local-coordinator",
        action="store_true",
        help="Use a paired coordinator on this Mac without an SSH tunnel",
    )

    coordinator = commands.add_parser("coordinator", help="Run the local coordinator")
    coordinator.add_argument(
        "--port", type=int, help="Listening port; defaults to DIN_PORT or 8000"
    )
    coordinator.add_argument("--strategy", choices=("round_robin", "capacity_aware"))

    provider = commands.add_parser("provider", help="Run one independent local provider")
    provider.add_argument("--port", type=int, help="Listening port; defaults to DIN_PORT or 8101")
    provider.add_argument("--provider-id")
    provider.add_argument("--coordinator-url")

    demo = commands.add_parser("demo", help="Supervise a coordinator and two provider processes")
    demo.add_argument("--coordinator-port", type=int, default=8000)
    demo.add_argument("--provider-ports", type=int, nargs=2, default=(8101, 8102))
    demo.add_argument("--startup-timeout", type=float, default=15)
    demo.add_argument("--strategy", choices=("round_robin", "capacity_aware"))
    return parser


def main() -> int:
    configure_logging()
    parser = _parser()
    arguments = parser.parse_args()
    strategy = getattr(arguments, "strategy", None)
    if strategy is not None:
        os.environ["DIN_STRATEGY"] = strategy

    try:
        if arguments.command in (None, "interactive"):
            from inference_network.interactive import run_interactive

            lan = getattr(arguments, "lan", False)
            profile = getattr(arguments, "profile", None)
            local = getattr(arguments, "local_coordinator", False)
            if (profile is not None or local) and not lan:
                raise ValueError("Pairing options require explicit --lan mode")
            return run_interactive(lan, profile, local)
        if arguments.command == "demo":
            return run_network(
                coordinator_port=arguments.coordinator_port,
                provider_ports=tuple(arguments.provider_ports),
                startup_timeout=arguments.startup_timeout,
            )

        if arguments.port is not None:
            os.environ["DIN_PORT"] = str(arguments.port)
        settings: Settings
        if arguments.command == "provider":
            for field in ("provider_id", "coordinator_url"):
                value = getattr(arguments, field)
                if value is not None:
                    os.environ[f"DIN_{field.upper()}"] = value
            settings = ProviderSettings.from_env()
        else:
            settings = CoordinatorSettings.from_env()

        uvicorn.run(
            f"inference_network.{arguments.command}:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            workers=1,
            access_log=False,
            timeout_graceful_shutdown=3,
        )
        return 0
    except ValidationError as error:
        fields = sorted(
            {
                ".".join(str(part) for part in issue["loc"]) or "settings"
                for issue in error.errors(
                    include_url=False, include_input=False, include_context=False
                )
            }
        )
        print(f"Startup failed: invalid configuration for {', '.join(fields)}", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
