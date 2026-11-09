import os

import pytest

URL = os.environ.get("JOSHUA_URL", "http://127.0.0.1:18080")
MODEL = os.environ.get("JOSHUA_MODEL", "tiny")


@pytest.fixture(scope="session")
def url():
    return URL


@pytest.fixture(scope="session")
def model():
    return MODEL
