import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import matplotlib.pyplot as plt


THRESHOLD_SEARCH = True
LOSS_WEIGHT = True
POS_W = 0.86 / 0.14


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


def load_data(data_path, ratios, scaler=None):
    df = pd.read_csv(data_path)
    train_ratio, valid_ratio, test_ratio = ratios
    temp, test = train_test_split(df, test_size=test_ratio, random_state=1)
    train, valid = train_test_split(
        temp, test_size=valid_ratio / (train_ratio + valid_ratio), random_state=1
    )

    train.to_csv("./data/train.csv", index=False)
    valid.to_csv("./data/valid.csv", index=False)
    test.to_csv("./data/test.csv", index=False)

    target_col = "Diabetes_binary"
    x_train, y_train = train.drop(target_col, axis=1).values, train[target_col].values
    x_valid, y_valid = valid.drop(target_col, axis=1).values, valid[target_col].values
    x_test, y_test = test.drop(target_col, axis=1).values, test[target_col].values

    if scaler:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

    # Converting to Tensors objects
    x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(
        y_train, dtype=torch.float
    )
    x_valid, y_valid = torch.tensor(x_valid, dtype=torch.float), torch.tensor(
        y_valid, dtype=torch.float
    )
    x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(
        y_test, dtype=torch.float
    )

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def main():
    # Preparing data
    ratios = (0.7, 0.15, 0.15)
    path_data = "data.csv"
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(
        path_data, ratios, scaler=None
    )

    # Define key-variables
    learning_rate = 0.001
    num_feat = x_train.shape[1]
    epochs = 1000
    pos_weight = torch.tensor(POS_W, dtype=torch.float) if LOSS_WEIGHT else None
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model, optimizer = get_model(num_feat, learning_rate)
    train_ds, valid_ds = (x_train, y_train), (x_valid, y_valid)
    train_ls, valid_ls = fit(
        epochs,
        model,
        loss_func,
        optimizer,
        train_ds,
        valid_ds,
        penalty=None,
    )
    # Saving the model
    torch.save(model.state_dict(), "msd.pt")

    # Plotting the losses result
    indices = list(range(epochs))
    plt.plot(indices, train_ls, label="Train Loss", color="blue")
    plt.plot(indices, valid_ls, label="Valid Loss", color="red")
    plt.legend(loc="best")
    plt.savefig("loss.png")
    plt.close()
    # plt.show()

    # Print final loss of train,valid, test
    print("********** Report **************")
    print(f"Train final Loss: {train_ls[-1]}")
    print(f"Valid final Loss: {valid_ls[-1]}")
    # print(f'Test final loss: {loss_func(model(x_test), y_test)}')

    # Printing the classifications results
    valid_pred = model.pred(x_valid)
    print("************* Validation Report *************")
    print(classification_report(y_valid, valid_pred))

    # Print trained model weights and bias
    print(
        f"Model Weights: {[round(w, 5) for w in model.lin.weight.squeeze().tolist()]}"
    )
    print(f"Model bias: {round(model.lin.bias.item(), 5)}")

    # Search for optimal threshold
    if THRESHOLD_SEARCH:
        thresholds = np.arange(0, 0.81, 0.01)
        f1 = []
        for thr in thresholds:
            valid_pred = model.pred(x_valid, threshold=thr)
            f1.append(f1_score(y_valid, valid_pred, average="binary"))

        plt.plot(thresholds, f1, label="F1_Score")
        plt.legend(loc="best")
        plt.savefig("thre_search.png")
        # plt.show()

        best_thr = thresholds[np.argmax(f1)]
        valid_pred = model.pred(x_valid, threshold=best_thr)
        print("************* Validation Report-Threshold Optimized *************")
        print(f"best threshold for validation: {best_thr}")
        print(classification_report(y_valid, valid_pred))

    # test_pred = model.pred(x_test, best_thr if THRESHOLD_SEARCH else 0.5)
    # print('************* Test Report *************')
    # print(classification_report(y_test, test_pred))


if __name__ == "__main__":
    main()
