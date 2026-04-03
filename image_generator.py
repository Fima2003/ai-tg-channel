from __future__ import annotations

import colorsys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from config import BotConfig
from models import ContentItem, GeneratedBatch


@dataclass(slots=True)
class ImageGenerationResult:
    item_index: int
    image_path: Path


def _gradient_color(seed: int, offset: float) -> tuple[int, int, int]:
    hue = ((seed % 360) / 360.0 + offset) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _wrap(text: str, width: int = 28) -> str:
    paragraphs = [
        textwrap.fill(part.strip(), width=width)
        for part in text.split("\n")
        if part.strip()
    ]
    return "\n".join(paragraphs) if paragraphs else text


def _render_image(item: ContentItem, output_path: Path, batch_title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1080, 1080), _gradient_color(item.index * 47, 0.15))
    draw = ImageDraw.Draw(image)

    accent = _gradient_color(item.index * 83, 0.55)
    draw.rectangle((54, 54, 1026, 1026), outline=accent, width=8)
    draw.rectangle((96, 96, 984, 984), outline=(255, 255, 255), width=2)

    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    title_text = _wrap(batch_title.upper(), width=18)
    caption_text = _wrap(item.caption, width=26)
    prompt_text = _wrap(item.prompt, width=34)

    draw.multiline_text(
        (120, 160), title_text, fill=(255, 255, 255), font=title_font, spacing=8
    )
    draw.multiline_text(
        (120, 320), caption_text, fill=(245, 245, 245), font=body_font, spacing=8
    )
    draw.multiline_text(
        (120, 560), prompt_text, fill=(235, 235, 235), font=body_font, spacing=6
    )
    draw.text((120, 920), f"Image {item.index}", fill=(255, 255, 255), font=body_font)

    image.save(output_path)


async def generate_images(
    config: BotConfig, batch: GeneratedBatch
) -> list[ImageGenerationResult]:
    results: list[ImageGenerationResult] = []
    for item in batch.items:
        _render_image(item, item.image_path, batch.title)
        results.append(
            ImageGenerationResult(item_index=item.index, image_path=item.image_path)
        )
    return results
