import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parent / "fixtures/prepare.py"
SPEC = importlib.util.spec_from_file_location("fixture_prepare", SCRIPT)
assert SPEC and SPEC.loader
fixture_prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_prepare)


def test_suite_fixture_ids_follow_the_golden_inventory() -> None:
    assert fixture_prepare.fixture_ids("golden-large") == (
        "perf_junk_shop_1280x720",
        "perf_blender_classroom_1280x720",
    )
    assert fixture_prepare.fixture_ids("performance-large") == (
        "perf_junk_shop_1280x720",
    )
    assert fixture_prepare.fixture_ids("performance-small") == (
        "perf_junk_shop_1280x720",
    )

    with pytest.raises(ValueError, match="no fixture preparation contract"):
        fixture_prepare.fixture_ids("unknown")
