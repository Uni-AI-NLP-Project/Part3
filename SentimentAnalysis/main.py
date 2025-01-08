import pandas as pd
import numpy as np
import _pickle as pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


PATT = r"\S+"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "eval"
SAMPLE_SIZE = None
DATA_PATH = "./data/twitter.csv"
MIN_WORD_FREQ = 3
MODEL_FN = "sample"
CLASSES = 3
EPOCHS = 10000


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


def fit(EPOCHS, model, loss_func, opt, train, valid) -> tuple[list, list]:
    x_train, y_train = train
    x_valid, y_valid = valid
    train_losses, valid_losses = [], []
    for epoch in range(EPOCHS):
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


def create_vectorizer(train_data) -> CountVectorizer:
    cv = CountVectorizer(min_df=MIN_WORD_FREQ, binary=True, token_pattern=PATT)
    cv.fit(train_data)
    return cv


def save_model_and_vectorizer(model, vectorizer, filename):
    """Save both the model state and the vectorizer."""
    # Save model state
    torch.save(model.state_dict(), f"models/{filename}.pt")

    # Save vectorizer using pickle
    with open(f"models/{filename}_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, file=f)


def load_model_and_vectorizer(model_filename):
    """Load both the model state and the vectorizer."""

    # Load vectorizer
    with open(f"./models/{model_filename}_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Create model with correct input size
    model = LogisticRegression(len(vectorizer.vocabulary_), 3).to(DEVICE)

    # Load model state
    model.load_state_dict(
        torch.load(f"./models/{model_filename}.pt", weights_only=True)
    )

    return model, vectorizer


def load_data():
    train = pd.read_csv("./data/train.csv")
    valid = pd.read_csv("./data/valid.csv")
    test = pd.read_csv("./data/test.csv")

    return train, valid, test


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


def df2tensors(df: pd.DataFrame, cv: CountVectorizer):
    x = df.iloc[:, 0]
    x = cv.transform(x).toarray()  # type: ignore
    x = torch.tensor(x, dtype=torch.float).to(DEVICE)
    y = torch.tensor(df.iloc[:, 1].values, dtype=torch.long).to(DEVICE)
    return x, y


def save_plots(train_ls, valid_ls):
    plt.figure(figsize=(16, 9))
    plt.plot(train_ls, label="Train Loss", color="blue")
    plt.plot(valid_ls, label="Valid Loss", color="red")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"./plots/loss_{len(train_ls)}_EPOCHS.png")
    plt.close()


def main():
    # Load the raw data
    if SAMPLE_SIZE:
        df = pd.read_csv(DATA_PATH)
        df = df.sample(SAMPLE_SIZE, random_state=1)
        train_df, valid_df, test_df = split_data(df, (0.6, 0.2, 0.2))
    else:
        train_df, valid_df, test_df = load_data()

    print(f"Using {DEVICE} device")

    if MODE == "train":
        # preprocess it with CountVectorizer
        cv = create_vectorizer(train_df.iloc[:, 0])
        x_train, y_train = df2tensors(train_df, cv)
        x_valid, y_valid = df2tensors(valid_df, cv)

        # Create Model
        num_features = len(cv.vocabulary_)
        lr = 0.001
        model, opt = get_model(num_features, CLASSES)

        # Train the model(s)
        criteria = nn.CrossEntropyLoss()

        # Pack the train and valid
        train = (x_train, y_train)
        valid = (x_valid, y_valid)

        train_losses, valid_losses = fit(EPOCHS, model, criteria, opt, train, valid)
        # Save plots
        save_plots(train_losses, valid_losses)
        # Save them
        model_fname = (
            f"model_sample:_{SAMPLE_SIZE},{EPOCHS}_EPOCHS" if SAMPLE_SIZE else "model0"
        )

        save_model_and_vectorizer(model, cv, model_fname)
        # Evaluate it with the test data
    elif MODE == "eval":
        with torch.inference_mode():
            models_fnames = [f"model_sample:_{SAMPLE_SIZE},{EPOCHS}_EPOCHS"]
            for fname in models_fnames:
                model, cv_loaded = load_model_and_vectorizer(fname)
                x_train, y_train = df2tensors(train_df, cv_loaded)
                x_valid, y_valid = df2tensors(valid_df, cv_loaded)
                x_test, y_test = df2tensors(test_df, cv_loaded)

                print(f"********** {fname.upper()} Performance *******")
                print("For Train dataset:")

                train_pred = model.pred(x_train)
                print(classification_report(y_train, train_pred, zero_division=0))
                print(confusion_matrix(y_train, train_pred))

                print(
                    f"Mean F1_Score: {f1_score(y_train, train_pred, average='macro')}"
                )
                print("For Validation dataset:")
                val_pred = model.pred(x_valid)
                print(classification_report(y_valid, val_pred, zero_division=0))
                print(f"Mean F1_Score: {f1_score(y_valid,val_pred, average='macro')}")
                print("For Test dataset:")
                test_pred = model.pred(x_test)
                print(classification_report(y_test, test_pred, zero_division=0))
                print(f"Mean F1_Score: {f1_score(y_test,test_pred, average='macro')}")


if __name__ == "__main__":
    main()
