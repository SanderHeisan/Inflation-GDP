import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import data_bundle  # noqa: E402
from usmodel import data_bundle as us_data_bundle  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return data_bundle.make_demo_bundle(seed=7)


@pytest.fixture(scope="session")
def bundle_us():
    return us_data_bundle.make_demo_bundle(seed=11)
