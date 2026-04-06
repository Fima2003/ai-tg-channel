from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from config import BotConfig
from models import ContentItem, GeneratedBatch


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageGenerationResult:
    item_index: int
    image_path: Path


def _response_get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _job_result(response: Any) -> Any:
    jobs = _response_get(response, "jobs", []) or []
    if not jobs:
        return None
    result = _response_get(jobs[0], "result", None)
    # Handle result as a list (actual SDK behavior) or dict
    if isinstance(result, list) and result:
        return result[0]
    return result


def _job_available(response: Any) -> bool:
    result = _job_result(response)
    if result is None:
        return False
    return bool(_response_get(result, "available", False))


def _blob_url(response: Any) -> str | None:
    result = _job_result(response)
    if result is None:
        return None
    # Check for blobUrl first (returned when available=True)
    for key in ("blobUrl", "imageUrl", "url"):
        value = _response_get(result, key)
        if value:
            return str(value)
    # No blob URL available; job may still be processing
    return None


def _job_identity(response: Any) -> tuple[str | None, str | None]:
    token = _response_get(response, "token")
    jobs = _response_get(response, "jobs", []) or []
    job_id = _response_get(jobs[0], "jobId") if jobs else None
    return (str(token) if token else None, str(job_id) if job_id else None)


def _guess_extension(blob_url: str) -> str:
    suffix = Path(blob_url.split("?", 1)[0]).suffix
    return suffix if suffix else ".png"


def _download_blob(blob_url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(blob_url) as response:
        output_path.write_bytes(response.read())


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _generate_civitai_image(
    config: BotConfig, item: ContentItem, batch: GeneratedBatch
) -> Path:
    try:
        import civitai
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Civitai SDK is not installed. Install the project dependencies first."
        ) from exc

    seed = batch.run_date.toordinal() * 100 + item.index
    options = {
        "model": config.civitai_model,
        "params": {
            "prompt": item.prompt,
            "negativePrompt": config.civitai_negative_prompt,
            "scheduler": config.civitai_scheduler,
            "steps": config.civitai_steps,
            "cfgScale": config.civitai_cfg_scale,
            "width": config.civitai_width,
            "height": config.civitai_height,
            "seed": seed,
            "clipSkip": config.civitai_clip_skip,
        },
    }

    logger.info("Submit Civitai job for item %s", item.index)
    # Avoid SDK wait-mode internals; poll job status ourselves for stability.
    response = await _maybe_await(civitai.image.create(options, wait=False))
    logger.debug("Initial response: available=%s", _job_available(response))

    blob_url = _blob_url(response)
    if not blob_url or not _job_available(response):
        token, job_id = _job_identity(response)
        token_preview = token[:20] if token else None
        logger.info(
            "Job submitted: token=%s, job_id=%s, awaiting completion",
            token_preview,
            job_id,
        )
        for attempt in range(config.civitai_poll_attempts):
            if not token and not job_id:
                logger.warning("No token or job_id to poll")
                break
            logger.debug(
                "Polling Civitai job %s for item %s (%s/%s)",
                job_id or token_preview,
                item.index,
                attempt + 1,
                config.civitai_poll_attempts,
            )
            await asyncio.sleep(config.civitai_poll_interval)
            response = await _maybe_await(civitai.jobs.get(token=token, job_id=job_id))
            available = _job_available(response)
            logger.debug("Poll attempt %d: available=%s", attempt + 1, available)
            if available:
                blob_url = _blob_url(response)
                if blob_url:
                    logger.info("Image ready at attempt %d", attempt + 1)
                    break

    if not blob_url:
        raise RuntimeError("Civitai did not return an image URL for the job")

    output_path = item.image_path
    guessed_extension = _guess_extension(blob_url)
    if output_path.suffix != guessed_extension:
        output_path = output_path.with_suffix(guessed_extension)

    logger.info("Download Civitai image for item %s to %s", item.index, output_path)
    await asyncio.to_thread(_download_blob, blob_url, output_path)
    return output_path


async def generate_images(
    config: BotConfig, batch: GeneratedBatch
) -> list[ImageGenerationResult]:
    results: list[ImageGenerationResult] = []
    for item in batch.items:
        image_path = await _generate_civitai_image(config, item, batch)
        results.append(
            ImageGenerationResult(item_index=item.index, image_path=image_path)
        )
    return results
