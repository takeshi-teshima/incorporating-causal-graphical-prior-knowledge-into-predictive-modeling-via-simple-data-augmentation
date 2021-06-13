import numpy as np
import pandas as pd
import lingam

# Type hinting
from causal_data_augmentation.api_support.typing import GraphType


def estimate_lingam(X: pd.DataFrame) -> GraphType:
    model = lingam.DirectLiNGAM()
    model.fit(X)
    vertices = X.columns
    directed_edges = []
    bi_edges = []
    for i, j in zip(*np.nonzero(model.adjacency_matrix_)):
        directed_edges.append((vertices[j], vertices[i]))
    return vertices, directed_edges, bi_edges
