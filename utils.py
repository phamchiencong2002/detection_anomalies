""" Reusable helpers functions for the project. """
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import pandas as pd
import numpy as np

RANDOM_STATE = 42

def library_versions():
    packages = ["numpy", 
                "pandas",
                "scikit-learn",
                "matplotlib",
                "seaborn"
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return pd.Series(versions, name="version")

def load_mouse(path):
    """Load Mouse and construct evaluation labels b4 any shuffling.
    The assignment identifies the last 10 observations as anomalies.
    These labels must not be used to fit the outlier detectors or tune the label-free threshold heuristics."""

    X = np.loadtxt(Path(path))
    if X.shape != (500, 2):
        raise ValueError(f"Expected shape (500, 2), instead, received {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Mouse contains missing or infinite values.")

    y = np.zeros(len(X), dtype = int)
    y[-10:] = 1

    return X, y