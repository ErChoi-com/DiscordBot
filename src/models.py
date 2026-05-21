from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ScrapedItem:
    title: str
    link: str
    source_url: str
    item_type: str = "link"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "link": self.link,
            "source_url": self.source_url,
            "type": self.item_type,
        }
