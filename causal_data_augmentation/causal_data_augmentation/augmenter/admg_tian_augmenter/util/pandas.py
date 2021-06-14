import numpy as np
import pandas as pd


def repeat_df(df: pd.DataFrame, times: int) -> pd.DataFrame:
    """Repeat a DataFrame vertically and cyclically.

    Parameters:
        df : DataFrame to be repeated.
        times : The number of times to repeat ``df``.

    Returns:
        New DataFrame whose rows are the repeated ``df``.

    Example:
        >>> repeat_df(pd.DataFrame.from_dict([{'A': 1}, {'A': 2}]), 2)
           A
        0  1
        1  2
        2  1
        3  2
    """
    return df.append([df] * (times - 1), ignore_index=True)


def tile_df(df: pd.DataFrame, times: int) -> pd.DataFrame:
    """Tile a DataFrame vertically by repeating the rows contiguously.

    Parameters:
        df : DataFrame to be repeated.
        times : The number of times to repeat each element of ``df``.

    Returns:
        New DataFrame whose rows are the repeated elements of ``df``.

    Example:
        >>> tile_df(pd.DataFrame.from_dict([{'A': 1}, {'A': 2}]), 2)
           A
        0  1
        0  1
        1  2
        1  2
    """
    return df.iloc[np.arange(len(df)).repeat(times)]


def product_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Take the Cartesian product of the rows of the two DataFrames.

    Parameters:
        df1 : The first DataFrame.
        df1 : The second DataFrame.

    Returns:
        New DataFrame.

    Example:
        >>> A = pd.DataFrame.from_dict([{'A': 1}, {'A': 2}])
        >>> B = pd.DataFrame.from_dict([{'B': 1}, {'B': 2}])
        >>> product_df(A, B)
           A  B
        0  1  1
        1  1  2
        2  2  1
        3  2  2
    """
    A = tile_df(df1, len(df2)).reset_index(drop=True)
    B = repeat_df(df2, len(df1)).reset_index(drop=True)
    return pd.concat((A, B), axis=1)


def summarize_duplicates_df(df: pd.DataFrame, key: str, method:str='sum') -> pd.DataFrame:
    """Perform contraction of duplicating entries while aggregating one of the columns.

    Parameters:
        df : DataFrame to summarize.
        key : The key to be aggregated.
        method : Method to use for aggregating the ``key`` column.

    Returns:
        New DataFrame.

    Example:
        >>> A = pd.DataFrame.from_dict([{'A': 1}, {'A': 1}, {'A': 2}, {'A': 2}])
        >>> B = pd.DataFrame.from_dict([{'B': 1}, {'B': 3}, {'B': 5}, {'B': 7}])
        >>> df = pd.concat((A, B), axis=1)
        >>> df
           A  B
        0  1  1
        1  1  3
        2  2  5
        3  2  7
        >>> summarize_duplicates_df(df, 'B', 'sum')
           A   B
        0  1   4
        2  2  12
    """
    cols = list(df.columns)
    cols.remove(key)
    df[key] = df.groupby(cols)[key].transform(method)
    return df.drop_duplicates()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
