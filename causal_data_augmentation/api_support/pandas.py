import pandas as pd


def df_difference(a, b):
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    df = pd.merge(a, b, indicator=True, how='outer')
    left = df.query('_merge=="left_only"').drop('_merge', axis=1)
    right = df.query('_merge=="right_only"').drop('_merge', axis=1)
    both = df.query('_merge=="both"').drop('_merge', axis=1)
    return left, right, both
