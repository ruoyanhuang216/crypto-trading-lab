"""Abstract base class for all market signals."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseSignal(ABC):
    """Base class for all market signals.

    Subclasses implement `compute`, which takes an OHLCV DataFrame and
    returns a copy with indicator columns appended. The primary output
    column is declared in `output_col`.

    Unlike strategies, signals do not emit position decisions — they
    describe market conditions (trend strength, volatility regime, etc.)
    that strategies can consume as inputs.
    """

    def __init__(self, **params):
        self.params = params

    @property
    @abstractmethod
    def output_col(self) -> str:
        """Name of the primary output column added by compute()."""
        ...

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute signal columns and return an augmented copy of df.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (UTC) and columns
                open, high, low, close, volume.

        Returns:
            Copy of df with signal columns appended.
        """
        ...

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"
