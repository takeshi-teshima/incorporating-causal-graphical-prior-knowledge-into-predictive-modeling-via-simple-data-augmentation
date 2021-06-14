import numpy as np
import pandas as pd


def repeat_df(df, times):
    """
    Example:
        >>> repeat_df(pd.DataFrame.from_dict([{'A': 1}, {'A': 2}]), 2)
           A
        0  1
        1  2
        2  1
        3  2
    """
    return df.append([df] * (times - 1), ignore_index=True)


def tile_df(df, times):
    """
    Example:
        >>> tile_df(pd.DataFrame.from_dict([{'A': 1}, {'A': 2}]), 2)
           A
        0  1
        0  1
        1  2
        1  2
    """
    return df.iloc[np.arange(len(df)).repeat(times)]


def product_df(df1, df2):
    """
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


def summarize_duplicates_df(df, key, method='sum'):
    """
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
