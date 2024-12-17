import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lin = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.lin(x).sigmoid().view(-1)

    def pred(self, x, threshold: float = 0.5):
        y_pred = self.forward(x)
        return torch.where(y_pred > threshold, 1, 0)


def get_model(n_features, lr: float = 0.01):
    model = LogisticRegression(n_features)
    return model, SGD(model.parameters(), lr)


def fit(epochs, model, loss_func, opt, train, valid) -> tuple[list, list]:
    x_train, y_train = train
    x_valid, y_valid = valid
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        train_pred = model(x_train)
        train_loss = loss_func(train_pred, y_train)
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            valid_loss = loss_func(model(x_valid), y_valid)
            valid_losses.append(valid_loss.item())

        model.train()
        opt.zero_grad()
        train_loss.backward()
        opt.step()

    return train_losses, valid_losses


def main():
    # Splitting data
    df = pd.read_csv('data.csv')
    train_ratio, valid_ratio, test_ratio = .7, .15, .15
    temp, test = train_test_split(df, test_size=test_ratio, random_state=1)
    train, valid = train_test_split(temp, test_size=valid_ratio / (train_ratio + valid_ratio), random_state=1)

    # Converting to Tensors objects
    target_col = 'Diabetes_binary'
    x_train, y_train = (torch.FloatTensor(train.drop(target_col, axis=1).values),
                        torch.FloatTensor(train[target_col].values))
    x_valid, y_valid = (torch.FloatTensor(valid.drop(target_col, axis=1).values),
                        torch.FloatTensor(valid[target_col].values))
    x_test, y_test = (torch.FloatTensor(test.drop(target_col, axis=1).values),
                      torch.FloatTensor(test[target_col].values))

    # Define key-variables
    learning_rate = 0.01
    num_feat = x_train.shape[1]
    epochs = 1000

    model, optimizer = get_model(num_feat, learning_rate)
    train_ds, valid_ds = (x_train, y_train), (x_valid, y_valid)
    train_ls, valid_ls = fit(epochs, model, F.binary_cross_entropy, optimizer, train_ds, valid_ds)

    # Plotting the losses result
    indices = list(range(epochs))
    plt.plot(indices, train_ls, label='Train Loss', color='blue')
    plt.plot(indices, valid_ls, label='Valid Loss', color='red')
    plt.legend(loc='best')
    plt.show()

    # Print final loss of train,valid, test
    print(f'Train final Loss: {train_ls[-1]}')
    print(f'Valid final Loss: {valid_ls[-1]}')
    print(f'Test final loss: {F.binary_cross_entropy(model(x_test), y_test)}')

    # Printing the classifications results
    valid_pred = model.pred(x_valid)
    print('************* Validation Report *************')
    print(classification_report(y_valid, valid_pred))
    test_pred = model.pred(x_test)
    print('************* Test Report *************')
    print(classification_report(y_test, test_pred))

    # Print trained model weights and bias
    print(f'Model Weights: {model.lin.weight.data}')
    print(f'Model bias: {model.lin.bias.item()}')


if __name__ == '__main__':
    main()
