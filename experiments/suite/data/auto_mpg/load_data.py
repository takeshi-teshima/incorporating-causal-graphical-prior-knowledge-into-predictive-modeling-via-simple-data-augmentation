import pandas as pd

# Type hinting
from causal_data_augmentation.api_support.typing import GraphType

COLUMNS = [
    'mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
    'model year', 'origin', 'car name'
]

VARIABLES_REMOVED = ['origin', 'car name']


def get_predicted_variable_name():
    return 'mpg'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    data = pd.read_csv(path,
                       names=COLUMNS,
                       delim_whitespace=True,
                       na_values='?')
    data = data.dropna()
    data = data.drop(VARIABLES_REMOVED, axis=1)
    return data


if __name__ == '__main__':
    data = load_data('auto-mpg.data')
