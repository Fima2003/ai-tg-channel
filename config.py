from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


DEFAULT_POST_TIME: Final = "09:00"
DEFAULT_TIMEZONE: Final = "UTC"
DEFAULT_MIN_IMAGES: Final = 1
DEFAULT_MAX_IMAGES: Final = 4
DEFAULT_OLLAMA_HOST: Final = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: Final = "gemma4:e4b"
DEFAULT_STATE_FILE: Final = ".state/last_run.json"
DEFAULT_OUTPUT_DIR: Final = ".generated"
DEFAULT_MAX_RETRIES: Final = 3


@dataclass(slots=True)
class BotConfig:
    telegram_api_id: int
    telegram_api_hash: str
    telegram_bot_token: str
    telegram_channel: str | int
    timezone: str = DEFAULT_TIMEZONE
    post_time: str = DEFAULT_POST_TIME
    min_images: int = DEFAULT_MIN_IMAGES
    max_images: int = DEFAULT_MAX_IMAGES
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    state_file: Path = Path(DEFAULT_STATE_FILE)
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR)
    max_retries: int = DEFAULT_MAX_RETRIES


class ConfigurationError(ValueError):
    pass


def _parse_telegram_channel(value: str) -> str | int:
    channel = value.strip()

    # Accept private channel links like https://t.me/c/1234567890/42.
    if "t.me/c/" in channel:
        tail = channel.split("t.me/c/", 1)[1]
        raw_id = tail.split("/", 1)[0]
        if raw_id.isdigit():
            return int(f"-100{raw_id}")

    if channel.startswith("@"):
        return channel

    # Canonical private channel IDs are negative and start with -100.
    if channel.startswith("-100") and channel[4:].isdigit():
        return int(channel)

    # If the user pasted only the raw numeric part, normalize it.
    if channel.startswith("-") and channel[1:].isdigit():
        return int(f"-100{channel[1:]}")
    if channel.isdigit():
        return int(f"-100{channel}")
    return channel


def load_config(env_file: str | None = ".env") -> BotConfig:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    def required(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise ConfigurationError(f"Missing required environment variable: {name}")
        return value

    telegram_api_id = int(required("TELEGRAM_API_ID"))
    telegram_api_hash = required("TELEGRAM_API_HASH")
    telegram_bot_token = required("TELEGRAM_BOT_TOKEN")
    telegram_channel = _parse_telegram_channel(required("TELEGRAM_CHANNEL"))

    timezone = os.getenv("TIMEZONE", DEFAULT_TIMEZONE)
    post_time = os.getenv("POST_TIME", DEFAULT_POST_TIME)
    min_images = int(os.getenv("MIN_IMAGES", str(DEFAULT_MIN_IMAGES)))
    max_images = int(os.getenv("MAX_IMAGES", str(DEFAULT_MAX_IMAGES)))
    ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    state_file = Path(os.getenv("STATE_FILE", DEFAULT_STATE_FILE))
    output_dir = Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    max_retries = int(os.getenv("MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

    if min_images < 1:
        raise ConfigurationError("MIN_IMAGES must be at least 1")
    if max_images < min_images:
        raise ConfigurationError(
            "MAX_IMAGES must be greater than or equal to MIN_IMAGES"
        )
    if ":" not in post_time:
        raise ConfigurationError("POST_TIME must be in HH:MM format")
    hour, minute = post_time.split(":", 1)
    if not hour.isdigit() or not minute.isdigit():
        raise ConfigurationError("POST_TIME must contain numeric hour and minute")
    if not 0 <= int(hour) <= 23 or not 0 <= int(minute) <= 59:
        raise ConfigurationError("POST_TIME must be a valid 24-hour time")
    if max_retries < 0:
        raise ConfigurationError("MAX_RETRIES cannot be negative")

    return BotConfig(
        telegram_api_id=telegram_api_id,
        telegram_api_hash=telegram_api_hash,
        telegram_bot_token=telegram_bot_token,
        telegram_channel=telegram_channel,
        timezone=timezone,
        post_time=post_time,
        min_images=min_images,
        max_images=max_images,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        state_file=state_file,
        output_dir=output_dir,
        max_retries=max_retries,
    )
