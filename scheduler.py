from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from config import BotConfig
from content_generator import build_batch
from image_generator import generate_images
from models import BotState, GeneratedBatch, RunResult
from telegram_client import post_batch, resolve_target_entity


logger = logging.getLogger(__name__)


class DailyScheduler:
    def __init__(self, config: BotConfig, client) -> None:
        self.config = config
        self.client = client
        self.zone = ZoneInfo(config.timezone)
        self.state_file = config.state_file
        self.stop_event = asyncio.Event()

    def _load_state(self) -> BotState:
        if not self.state_file.exists():
            return BotState()
        data = json.loads(self.state_file.read_text(encoding="utf-8"))
        return BotState(
            last_success_date=data.get("last_success_date"),
            last_success_channel=data.get("last_success_channel"),
        )

    def _current_channel_key(self) -> str:
        return str(self.config.telegram_channel)

    def _was_successful_for(self, state: BotState, run_date: date) -> bool:
        return (
            state.last_success_date == run_date.isoformat()
            and state.last_success_channel == self._current_channel_key()
        )

    def _save_state(self, state: BotState) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(asdict(state), indent=2), encoding="utf-8"
        )

    def _scheduled_time(self) -> time:
        hour, minute = map(int, self.config.post_time.split(":"))
        return time(hour=hour, minute=minute, tzinfo=self.zone)

    def _next_run_at(self, now: datetime) -> datetime:
        current_time = self._scheduled_time()
        candidate = datetime.combine(now.date(), current_time, tzinfo=self.zone)
        if candidate <= now:
            candidate = datetime.combine(
                now.date() + timedelta(days=1), current_time, tzinfo=self.zone
            )
        return candidate

    async def _execute_once(
        self, run_date: date, persist_state: bool = True
    ) -> RunResult:
        state = self._load_state()
        if self._was_successful_for(state, run_date):
            return RunResult(
                run_date=run_date, success=True, skipped=True, detail="already posted"
            )

        entity = None
        if self.client is not None:
            # Fail fast if the bot cannot resolve or access the target channel.
            entity = await resolve_target_entity(self.client, self.config)

        batch = await build_batch(self.config, run_date)
        image_results = await generate_images(self.config, batch)
        if self.client is not None:
            await post_batch(self.client, self.config, batch, image_results, entity)

        if persist_state:
            self._save_state(
                BotState(
                    last_success_date=run_date.isoformat(),
                    last_success_channel=self._current_channel_key(),
                )
            )
        return RunResult(
            run_date=run_date, success=True, detail=f"posted {len(batch.items)} items"
        )

    async def run_once(
        self,
        run_date: date | None = None,
        retries: int | None = None,
        persist_state: bool = True,
    ) -> RunResult:
        run_date = run_date or datetime.now(self.zone).date()
        retries = self.config.max_retries if retries is None else retries
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await self._execute_once(run_date, persist_state=persist_state)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Run failed for %s (attempt %s/%s): %s",
                    run_date.isoformat(),
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                logger.debug("Run failure traceback", exc_info=True)
                if attempt >= retries:
                    break
                await asyncio.sleep(min(300, 2**attempt * 15))
        return RunResult(
            run_date=run_date,
            success=False,
            detail=str(last_error) if last_error else "unknown error",
        )

    async def run_forever(self) -> None:
        while not self.stop_event.is_set():
            now = datetime.now(self.zone)
            today_run_at = datetime.combine(
                now.date(), self._scheduled_time(), tzinfo=self.zone
            )
            state = self._load_state()

            if now >= today_run_at and not self._was_successful_for(state, now.date()):
                logger.info("Missed schedule; run now")
                result = await self.run_once(run_date=now.date())
                logger.info("Catch-up result: %s", result.detail)
                continue

            next_run = self._next_run_at(now)
            delay = max(0.0, (next_run - now).total_seconds())
            logger.info("Next run at %s", next_run.isoformat())
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                logger.info("Scheduled run start")
                result = await self.run_once(run_date=next_run.date())
                logger.info("Scheduled result: %s", result.detail)
                continue
            if self.stop_event.is_set():
                break

    def stop(self) -> None:
        self.stop_event.set()
