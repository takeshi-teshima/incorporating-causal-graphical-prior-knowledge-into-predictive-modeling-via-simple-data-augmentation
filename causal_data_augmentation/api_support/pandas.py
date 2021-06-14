import pandas as pd

# Type hinting
from typing import Tuple


def df_difference(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the difference of two DataFrames as sets.

    Parameters:
        a : First DataFrame
        b : Second DataFrame

    Returns:
        Tuple containing

        - ``left`` : Rows that appear only in ``a`` (i.e., left-only rows).
        - ``right`` : Rows that appear only in ``b`` (i.e., right-only rows).
        - ``both`` : Rows that appear both in ``a`` and ``b``.
    """
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    df = pd.merge(a, b, indicator=True, how='outer')
    left = df.query('_merge=="left_only"').drop('_merge', axis=1)
    right = df.query('_merge=="right_only"').drop('_merge', axis=1)
    both = df.query('_merge=="both"').drop('_merge', axis=1)
    return left, right, both
