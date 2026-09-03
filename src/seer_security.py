"""Explicit trust boundary for legacy local artifacts.

Pickle is executable input, not a safe interchange format. No trust is inferred
from a filename, directory, or a failed restricted checkpoint load.
"""

import os
import warnings


def require_trusted_pickle():
    if os.environ.get("SEER_ALLOW_UNSAFE_PICKLE") != "1":
        raise PermissionError(
            "Legacy Pickle loading is disabled. Only for artifacts you created "
            "locally or independently verified, set SEER_ALLOW_UNSAFE_PICKLE=1 "
            "for this process. Pickle can execute arbitrary code."
        )
    warnings.warn("Loading explicitly trusted legacy Pickle input; arbitrary code execution is possible.", RuntimeWarning, stacklevel=2)


def load_pickle(handle, *args, **kwargs):
    require_trusted_pickle()
    import dill
    return dill.load(handle, *args, **kwargs)


def read_pandas_pickle(path, *args, **kwargs):
    require_trusted_pickle()
    import pandas as pd
    return pd.read_pickle(path, *args, **kwargs)


def load_checkpoint(path, *, map_location=None):
    import torch
    # Never silently fall back to unrestricted unpickling.
    return torch.load(path, map_location=map_location, weights_only=True)
