from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from config import ConfigurationError, load_config
from scheduler import DailyScheduler
from telegram_client import create_client


def configure_logging() -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    debug_file = logging.FileHandler(Path("debug.log"), encoding="utf-8")
    debug_file.setLevel(logging.DEBUG)
    debug_file.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(debug_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily Telegram image bot")
    parser.add_argument(
        "--once", action="store_true", help="Run a single posting cycle and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate content without posting to Telegram",
    )
    parser.add_argument("--env-file", default=".env", help="Path to the .env file")
    return parser.parse_args()


@asynccontextmanager
async def maybe_client(config, dry_run: bool):
    if dry_run:
        yield None
        return
    client = await create_client(config)
    try:
        yield client
    finally:
        await client.disconnect()


async def async_main() -> int:
    args = parse_args()
    configure_logging()

    try:
        config = load_config(args.env_file)
    except ConfigurationError as exc:
        logging.error("Configuration error: %s", exc)
        return 2

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.state_file.parent.mkdir(parents=True, exist_ok=True)

    async with maybe_client(config, dry_run=args.dry_run) as client:
        scheduler = DailyScheduler(config, client if not args.dry_run else None)
        if args.dry_run:
            result = await scheduler.run_once(persist_state=False)
            logging.info("Dry-run result: %s", result)
            return 0 if result.success else 1
        if args.once:
            result = await scheduler.run_once()
            logging.info("Run result: %s", result)
            return 0 if result.success else 1
        await scheduler.run_forever()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
