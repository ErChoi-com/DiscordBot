# pyright: reportMissingImports=false

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.views import JobTextModal


class _DummyStore:
    def get_job_settings(self, channel_id: int) -> dict[str, object]:
        return {
            "keywords": "python developer",
            "location": "Canada",
            "radius_miles": 25,
            "results_wanted": 10,
        }


def test_job_text_modal_component_shape_is_stable() -> None:
    # Repeated instantiation should not accumulate extra modal components.
    for _ in range(20):
        modal = JobTextModal(store=_DummyStore(), channel_id=123)
        assert len(modal.children) == 5


def test_job_text_modal_labels_respect_discord_limit() -> None:
    modal = JobTextModal(store=_DummyStore(), channel_id=123)
    for child in modal.children:
        label = getattr(child, "label", "") or ""
        assert 1 <= len(label) <= 45
