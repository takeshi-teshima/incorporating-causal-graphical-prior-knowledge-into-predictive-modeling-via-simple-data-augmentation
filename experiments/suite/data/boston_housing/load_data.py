import pandas as pd

# Type hinting
from causal_data_augmentation.api_support.typing import GraphType

COLUMNS = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

VARIABLES_REMOVED = []


def get_predicted_variable_name():
    return 'MEDV'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    data = pd.read_csv(path, names=COLUMNS, delim_whitespace=True)
    data = data.dropna()
    data = data.astype({'RAD': float})
    data = data.drop(VARIABLES_REMOVED, axis=1)
    return data


if __name__ == '__main__':
    data = load_data('housing.data')
