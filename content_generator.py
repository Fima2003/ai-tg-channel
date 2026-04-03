from __future__ import annotations

import asyncio
import logging
import random
from datetime import date

from ollama import AsyncClient, ResponseError

from config import BotConfig
from models import ContentItem, GeneratedBatch


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You write short, vivid Telegram captions for a daily image post. "
    "Keep the tone fresh, concise, and concrete."
)


async def _safe_generate(host: str, model: str, prompt: str) -> str:
    try:
        client = AsyncClient(host=host)
        response = await client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.8,
                "top_p": 0.9,
            },
        )
        return response.message.content.strip()
    except (
        ResponseError,
        TimeoutError,
        OSError,
        asyncio.TimeoutError,
        ValueError,
    ) as exc:
        logger.warning("Ollama generation failed: %s", exc)
        return ""


async def build_batch(config: BotConfig, run_date: date) -> GeneratedBatch:
    rng = random.Random(run_date.toordinal())
    item_count = rng.randint(config.min_images, config.max_images)
    logger.info("Build content for %s (%d item)", run_date.isoformat(), item_count)

    # Keep post metadata deterministic; only item text uses the LLM.
    title = f"Daily drop for {run_date.strftime('%Y-%m-%d')}"
    intro_text = "Today's set is ready."

    items: list[ContentItem] = []
    for index in range(1, item_count + 1):
        logger.debug("Generate item text %d/%d", index, item_count)
        item_prompt = (
            "Write a detailed visual prompt for a stylized image. "
            f"Theme: {title}. Item {index} of {item_count}. Keep it concrete and visual."
        )
        caption_prompt = (
            "Write a short caption, under 20 words, for a Telegram channel image post. "
            f"Theme: {title}. Item {index} of {item_count}."
        )
        prompt_text = await _safe_generate(
            config.ollama_host, config.ollama_model, item_prompt
        )
        caption_text = await _safe_generate(
            config.ollama_host, config.ollama_model, caption_prompt
        )

        if not prompt_text:
            prompt_text = f"{title} - visual concept {index}"
        if not caption_text:
            caption_text = f"{title} #{index}"

        items.append(
            ContentItem(
                index=index,
                prompt=prompt_text,
                caption=caption_text,
                image_path=config.output_dir
                / run_date.isoformat()
                / f"item-{index}.png",
            )
        )

    return GeneratedBatch(
        run_date=run_date,
        title=title,
        intro_text=intro_text,
        items=items,
    )
