import functools
import inspect

import pytest
import torch

# For good practice, we *should* do this:
# See https://docs.pytest.org/en/stable/fixture.html#using-fixtures-from-other-projects
# pytest_plugins = ['e3nn.util.test']
# But doing so exposes float_tolerance to doctests, which don't support parametrized, autouse fixtures.
# Importing directly somehow only brings in the fixture later, preventing the issue.
from e3nn.util import test

# Suppress linter errors
float_tolerance = test.float_tolerance


@pytest.fixture
def load_trusted_module():
    """Load only whole-module checkpoints written by the requesting roundtrip test."""
    # Older supported Torch releases have no weights_only keyword.
    kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    return functools.partial(torch.load, **kwargs)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set the random seeds to try to get some reproducibility"""
    test.set_random_seeds()
