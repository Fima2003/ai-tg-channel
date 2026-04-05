from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from telethon import Button, TelegramClient, events, functions, types, utils
from telethon.errors import RPCError

from config import BotConfig
from image_generator import ImageGenerationResult
from models import GeneratedBatch


logger = logging.getLogger(__name__)

CALLBACK_PREFIX = "approve_join"


def _build_approve_callback_data(chat_id: int, user_id: int) -> bytes:
    return f"{CALLBACK_PREFIX}:{chat_id}:{user_id}".encode("ascii")


def _parse_approve_callback_data(data: bytes) -> tuple[int, int] | None:
    try:
        payload = data.decode("ascii")
    except UnicodeDecodeError:
        return None

    parts = payload.split(":")
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        return None

    try:
        chat_id = int(parts[1])
        user_id = int(parts[2])
    except ValueError:
        return None
    return chat_id, user_id


async def create_client(config: BotConfig) -> TelegramClient:
    client = TelegramClient(
        "ai-of-tg", config.telegram_api_id, config.telegram_api_hash
    )
    await client.start(bot_token=config.telegram_bot_token)
    await setup_join_request_handlers(client, config)
    return client


async def setup_join_request_handlers(
    client: TelegramClient, config: BotConfig
) -> None:
    target_entity = await resolve_target_entity(client, config)
    target_chat_id = utils.get_peer_id(target_entity)

    @client.on(events.Raw(types=types.UpdateBotChatInviteRequester))
    async def on_join_request(update: types.UpdateBotChatInviteRequester) -> None:
        request_chat_id = utils.get_peer_id(update.peer)
        if request_chat_id != target_chat_id:
            return

        callback_data = _build_approve_callback_data(request_chat_id, update.user_id)
        try:
            await client.send_message(
                update.user_id,
                "You are trying to subscribe to this channel.",
                buttons=[Button.inline("Subscribe now", data=callback_data)],
                link_preview=False,
            )
            logger.info(
                "Sent join verification DM to user %s for chat %s",
                update.user_id,
                request_chat_id,
            )
        except RPCError as exc:
            logger.warning(
                "Could not send join verification DM to user %s for chat %s: %s",
                update.user_id,
                request_chat_id,
                exc,
            )

    @client.on(events.CallbackQuery(pattern=rb"^approve_join:"))
    async def on_subscribe_now_click(event: events.CallbackQuery.Event) -> None:
        parsed = _parse_approve_callback_data(event.data)
        if parsed is None:
            await event.answer("Invalid request payload.", alert=True)
            return

        chat_id, user_id = parsed
        if event.sender_id != user_id:
            await event.answer("This button is not for your request.", alert=True)
            return

        try:
            await client(
                functions.messages.HideChatJoinRequestRequest(
                    peer=chat_id,
                    user_id=user_id,
                    approved=True,
                )
            )
        except RPCError as exc:
            logger.warning(
                "Failed to approve join request for user %s in chat %s: %s",
                user_id,
                chat_id,
                exc,
            )
            await event.answer(
                "Could not approve right now. Please request to join again.",
                alert=True,
            )
            return

        await event.edit(
            "Your request has been approved. You can now open the channel."
        )
        await event.answer("Approved")

    logger.info("Join request handlers are active for chat %s", target_chat_id)


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
