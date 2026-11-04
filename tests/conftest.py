"""Shared fixtures.

If Plotly is not installed (for example in a restricted sandbox), a minimal stand-in is
registered so that the data-preparation and validation logic of ``src.visualization`` can
still be tested. Tests marked ``needs_plotly`` are skipped in that situation, because a
stand-in cannot prove that a real figure renders. On a normal install the real Plotly is used.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

try:
    import plotly  # noqa: F401

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_PLOTLY = False
    _plotly = types.ModuleType("plotly")
    _plotly.__path__ = []
    _plotly.express = MagicMock(name="plotly.express")
    _plotly.graph_objects = MagicMock(name="plotly.graph_objects")
    sys.modules.update({"plotly": _plotly, "plotly.express": _plotly.express, "plotly.graph_objects": _plotly.graph_objects})


def pytest_collection_modifyitems(config, items):
    if HAS_PLOTLY:
        return
    skip = pytest.mark.skip(reason="plotly is not installed; figure rendering cannot be verified")
    for item in items:
        if "needs_plotly" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [20.0, 25.0, np.nan, 40.0, 35.0, 25.0],
        "city": ["A", "B", "B", None, "A", "B"],
        "score": [1, 2, 3, 4, 5, 2],
    })


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], n)
    y = (x1 + 0.5 * x2 + (group == "a") * 0.8 + rng.normal(0, 0.5, n) > 0.3).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "group": group, "label": np.where(y == 1, "yes", "no")})
    df.loc[rng.choice(n, 15, replace=False), "x1"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "group"] = np.nan
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 250
    a, b = rng.normal(size=n), rng.uniform(0, 10, n)
    kind = rng.choice(["u", "v"], n)
    y = 3 * a + 0.5 * b + (kind == "u") * 2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"a": a, "b": b, "kind": kind, "y": y})
    df.loc[rng.choice(n, 12, replace=False), "b"] = np.nan
    return df
