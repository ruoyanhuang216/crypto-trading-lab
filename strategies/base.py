"""Abstract base class that all strategies must inherit from."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Base class for all trading strategies.

    Subclasses must implement `generate_signals`, which takes an OHLCV
    DataFrame and returns it with an added 'signal' column:
      +1 = long
      -1 = short
       0 = flat / no position
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'signal' column to df and return the result.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (UTC) and columns
                open, high, low, close, volume.

        Returns:
            Copy of df with at least a 'signal' column added.
            Signal values: +1 (long), -1 (short), 0 (flat).
        """
        ...

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"
