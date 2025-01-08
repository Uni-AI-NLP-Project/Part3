import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
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


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)

    def forward(self, x):
        logits = self.lin(x)
        return logits

    def pred(self, x):
        with torch.no_grad():
            y_pred = self.forward(x)
            y_pred = torch.softmax(y_pred, dim=1)
            return y_pred.argmax(dim=1)


def get_model(n_features, n_classes, lr: float = 0.001):
    model = LogisticRegression(n_features, n_classes)
    return model, SGD(model.parameters(), lr)


def fit(
    epochs, model, loss_func, opt, train, valid, penalty=None, pen_lambda: float = 0.01
) -> tuple[list, list]:
    x_train, y_train = train
    x_valid, y_valid = valid
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        train_pred = model(x_train)
        train_loss = loss_func(train_pred, y_train)
        train_losses.append(train_loss.item())

        # Add regularization term
        if penalty:
            if penalty == "l1":
                train_loss += pen_lambda * torch.abs(model.lin.weight).sum()
            elif penalty == "l2":
                train_loss += pen_lambda / 2 * torch.pow(model.lin.weight, 2).sum()

        model.eval()
        with torch.no_grad():
            valid_loss = loss_func(model(x_valid), y_valid)
            valid_losses.append(valid_loss.item())

        model.train()
        opt.zero_grad()
        train_loss.backward()
        opt.step()

    return train_losses, valid_losses
