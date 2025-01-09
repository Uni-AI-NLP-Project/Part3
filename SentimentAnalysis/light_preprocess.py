import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # replace 'category' values from [-1,0,1] to [0,1,2]
    df.loc[:, "category"] += 1
    # remove data samples with na values
    df = df.dropna()
    return df


def split_data(
    df: pd.DataFrame, ratios: tuple[float, float, float], save: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ratio, valid_ratio, test_ratio = ratios
    temp, test = train_test_split(df, test_size=test_ratio, random_state=1)
    train, valid = train_test_split(
        temp, test_size=valid_ratio / (train_ratio + valid_ratio), random_state=1
    )
    if save:
        train.to_csv("./data/train.csv", index=False)
        valid.to_csv("./data/valid.csv", index=False)
        test.to_csv("./data/test.csv", index=False)
    return train, valid, test
