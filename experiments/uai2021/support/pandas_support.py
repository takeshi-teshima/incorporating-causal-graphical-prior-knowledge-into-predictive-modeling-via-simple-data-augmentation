import numpy as np
import pandas as pd


def match_df_idx(augmented_data, orig):
    matches = pd.DataFrame(columns=orig.columns)
    for col in orig.columns:
        match = (augmented_data[col].to_numpy()[:, None] ==
                 orig[col].to_numpy()[None, :]).nonzero()
        _, _idx = np.unique(match[0], return_index=True)
        matches[col] = match[1][_idx]
    return matches


def conditional_where(df, cond, where):
    return df.ply_where(~cond).append(df.ply_where(cond).ply_where(where))


def conditional_mutate(df, cond, expr):
    return df.ply_where(~cond).append(expr(df.ply_where(cond)))
