import bnlearn
import pandas as pd

# Type hinting
from causal_data_augmentation.api_support.typing import GraphType

VARIABLES_REMOVED = []


def get_predicted_variable_name():
    return 'PKA'


def load_data(path: str) -> pd.DataFrame:
    DATA_RENAME = {
        'raf': 'Raf',
        'mek': 'Mek',
        'plc': 'Plcg',
        'pip2': 'PIP2',
        'pip3': 'PIP3',
        'erk': 'Erk',
        'akt': 'Akt',
        'pka': 'PKA',
        'pkc': 'PKC',
        'p38': 'P38',
        'jnk': 'Jnk'
    }
    data = pd.read_csv(path, delimiter='\t')
    data = data.rename(columns=DATA_RENAME).drop(VARIABLES_REMOVED, axis=1)
    return data


def _bnlearn_adjmat_to_edge_tuples(adjmat: pd.DataFrame):
    edge_tuples = []
    for rowname in adjmat.index.values:
        for colname in adjmat.columns:
            if adjmat[colname][rowname]:
                edge_tuples.append((rowname, colname))
    return edge_tuples


def load_bif(path: str) -> GraphType:
    """
    Params:
        path : path to BIF file.
    """
    is_DAG = True
    verbose = 0
    bnlearn_model = bnlearn.import_DAG(path, CPD=is_DAG, verbose=verbose)
    bayesian_model, adjmat = bnlearn_model['model'], bnlearn_model['adjmat']
    adjmat = adjmat.drop(VARIABLES_REMOVED, axis=1).drop(VARIABLES_REMOVED,
                                                         axis=0)
    vertices = adjmat.columns
    directed_edges = _bnlearn_adjmat_to_edge_tuples(adjmat)
    bi_edges = []
    graph = vertices, directed_edges, bi_edges
    return graph


def load_consensus_graph(_=None) -> GraphType:
    """Load graph."""
    vertices = [
        'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC',
        'P38', 'Jnk'
    ]
    directed_edges = [('Plcg', 'PIP2'), ('Plcg', 'PKC'), ('PIP2', 'PKC'),
                      ('PIP3', 'PIP2'), ('PIP3', 'Plcg'), ('PIP3', 'Akt'),
                      ('PKA', 'Akt'), ('PKA', 'Erk'), ('PKA', 'Mek'),
                      ('PKA', 'Raf'), ('PKA', 'Jnk'), ('PKA', 'P38'),
                      ('PKC', 'Mek'), ('PKC', 'Raf'), ('PKC', 'Jnk'),
                      ('PKC', 'P38'), ('Mek', 'Erk')]
    bi_edges = []

    vertices = [v for v in vertices if v not in VARIABLES_REMOVED]
    directed_edges = [
        e for e in directed_edges
        if (e[0] not in VARIABLES_REMOVED) and (e[1] not in VARIABLES_REMOVED)
    ]
    bi_edges = [
        e for e in bi_edges
        if (e[0] not in VARIABLES_REMOVED) and (e[1] not in VARIABLES_REMOVED)
    ]
    graph = vertices, directed_edges, bi_edges
    return graph


def load_mooij_heskes_2013_graph(_=None) -> GraphType:
    """Load graph."""
    vertices = [
        'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC',
        'P38', 'Jnk'
    ]
    directed_edges = [
        ('PIP2', 'Plcg'),
        ('PIP3', 'PIP2'),
        ('Akt', 'Erk'),
        ('PKA', 'Akt'),
        ('PKA', 'Mek'),
        ('PKA', 'Jnk'),
        ('PKA', 'P38'),
        ('PKC', 'PKA'),
        ('PKC', 'Akt'),
        ('PKC', 'PIP2'),
        ('PKC', 'Plcg'),
        ('PKC', 'Mek'),
        ('PKC', 'Raf'),
        ('PKC', 'Jnk'),
        ('PKC', 'P38'),
        ('Mek', 'Raf'),
        ('Mek', 'Erk'),
    ]
    bi_edges = []

    vertices = [v for v in vertices if v not in VARIABLES_REMOVED]
    directed_edges = [
        e for e in directed_edges
        if (e[0] not in VARIABLES_REMOVED) and (e[1] not in VARIABLES_REMOVED)
    ]
    bi_edges = [
        e for e in bi_edges
        if (e[0] not in VARIABLES_REMOVED) and (e[1] not in VARIABLES_REMOVED)
    ]
    graph = vertices, directed_edges, bi_edges
    return graph


if __name__ == '__main__':
    data = load_data("main.result.ourvarrs/1. cd3cd28.txt")
    graph = load_bif("sachs.bif")
