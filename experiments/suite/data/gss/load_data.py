import pandas as pd

# Type hinting
from causal_data_augmentation.api_support.typing import GraphType


def get_predicted_variable_name():
    return 'v4'


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, delimiter=',')
    data = data.astype('float')
    return data


def load_graph(_=None) -> GraphType:
    """Load graph."""
    vertices = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    directed_edges = [('v1', 'v4'), ('v1', 'v5'), ('v3', 'v5'), ('v4', 'v2'),
                      ('v5', 'v2'), ('v5', 'v4'), ('v6', 'v4'), ('v6', 'v5')]
    bi_edges = [('v3', 'v1'), ('v1', 'v6'), ('v6', 'v3')]
    graph = vertices, directed_edges, bi_edges
    return graph


if __name__ == '__main__':
    data = load_data("testdata.csv")
    graph = load_graph()
