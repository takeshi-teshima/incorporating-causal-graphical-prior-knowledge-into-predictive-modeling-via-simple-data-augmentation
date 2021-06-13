import pandas as pd

# Type hinting
from causal_data_augmentation.api_support.typing import GraphType

VARIABLES_REMOVED = []


def get_predicted_variable_name():
    return 'quality'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    data = pd.read_csv(path, sep=';')
    # data = data.dropna()
    # data = data.astype('float')
    data = data.drop(VARIABLES_REMOVED, axis=1)
    return data


if __name__ == '__main__':
    data = load_data('winequality-red.csv'), load_data('winequality-white.csv')
