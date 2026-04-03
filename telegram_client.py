from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from telethon import TelegramClient

from config import BotConfig
from image_generator import ImageGenerationResult
from models import GeneratedBatch


logger = logging.getLogger(__name__)


async def create_client(config: BotConfig) -> TelegramClient:
    client = TelegramClient(
        "ai-of-tg", config.telegram_api_id, config.telegram_api_hash
    )
    await client.start(bot_token=config.telegram_bot_token)
    return client


async def resolve_target_entity(client: TelegramClient, config: BotConfig):
    logger.info("Resolving target channel")
    return await client.get_entity(config.telegram_channel)


async def post_batch(
    client: TelegramClient,
    config: BotConfig,
    batch: GeneratedBatch,
    image_results: Iterable[ImageGenerationResult],
    entity=None,
) -> None:
    if entity is None:
        entity = await resolve_target_entity(client, config)
    logger.info("Posting batch with %d image(s)", len(batch.items))

    for item, result in zip(batch.items, image_results, strict=False):
        await client.send_file(
            entity,
            result.image_path,
            caption=item.caption,
        )
