from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ContentItem:
    index: int
    prompt: str
    caption: str
    image_path: Path


@dataclass(slots=True)
class GeneratedBatch:
    run_date: date
    title: str
    intro_text: str
    items: list[ContentItem] = field(default_factory=list)


@dataclass(slots=True)
class BotState:
    last_success_date: Optional[str] = None
    last_success_channel: Optional[str] = None


@dataclass(slots=True)
class RunResult:
    run_date: date
    success: bool
    skipped: bool = False
    detail: str = ""
