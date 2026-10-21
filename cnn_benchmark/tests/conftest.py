import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import facility_mapping as fm  # noqa: E402
from build_parent_table import build  # noqa: E402
from common import load_config, load_processed  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def processed(cfg):
    return load_processed(cfg)


@pytest.fixture(scope="session")
def parent_table(cfg):
    return build(cfg)


@pytest.fixture(scope="session")
def reconstruction(cfg):
    return fm.reconstruct(cfg)


@pytest.fixture(scope="session")
def facilities(reconstruction):
    return reconstruction[0]


@pytest.fixture(scope="session")
def membership(reconstruction):
    return fm.parent_membership(reconstruction[1])
