import os
import pandas as pd
import numpy as np
import _pickle as pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from copy import deepcopy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "eval"
DATA_PATH = "./data/twitter.csv"

CLASSES = 3

# Vectorizer Parameters
PATT = r"\S+"
MIN_WORD_FREQ = 3

EPOCHS_PER_SAVE = 1000
SAVE_BEST = True
FIT_DEBUG = True

# Session Parameters
SID = 2  # session number IMPRTANT: CHANGE BETWEEN SCRIPT ACTIVATION!
SAMPLE_SIZE = None  # int: use part of the data. None: use all of the data
SRST = 5  # session random state
EPOCHS = 20000
SPATH = f"sessions/s{SID}"


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
    epochs, model: LogisticRegression, loss_func, opt, train, valid
) -> tuple[list, list]:
    x_train, y_train = train
    x_valid, y_valid = valid
    train_losses, valid_losses = [], []
    if SAVE_BEST:
        min_val_loss = float("inf")
        best_epoch = 0

    for epoch in range(epochs):
        # Training phase
        train_logits = model(x_train)
        train_loss = loss_func(train_logits, y_train)
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            valid_logits = model(x_valid)
            valid_loss = loss_func(valid_logits, y_valid).item()
            valid_losses.append(valid_loss)

            if SAVE_BEST and valid_loss < min_val_loss:
                best_model = deepcopy(model)
                min_val_loss = valid_loss
                best_epoch = epoch

            if EPOCHS_PER_SAVE and (epoch + 1) % EPOCHS_PER_SAVE == 0:
                save_model(model, f"{SPATH}/models/s{SID}e{epoch}")
                if FIT_DEBUG:
                    train_pred = model.pred(x_train)
                    valid_pred = model.pred(x_valid)
                    # Print loss, accuracy, and f1_score for train and validation
                    train_acc = accuracy_score(y_train, train_pred)
                    valid_acc = accuracy_score(y_valid, valid_pred)
                    train_macro_f1 = f1_score(y_train, train_pred, average="macro")
                    val_macro_f1 = f1_score(y_valid, valid_pred, average="macro")
                    print(
                        f"Epoch {epoch}: Train Loss = {train_loss:.5f}, Val Loss = {valid_loss:.5f} | "
                        f"Train Acc = {train_acc:.5f}, Val Acc = {valid_acc:.5f} | Train F1 = {train_macro_f1:.5f}, "
                        f"Val F1 = {val_macro_f1:.5f}"
                    )

        model.train()
        opt.zero_grad()
        train_loss.backward()
        opt.step()

    if SAVE_BEST:
        save_model(best_model, f"{SPATH}/models/model_best_epoch_{best_epoch}")

    return train_losses, valid_losses


def create_vectorizer(train_data) -> CountVectorizer:
    cv = CountVectorizer(min_df=MIN_WORD_FREQ, binary=True, token_pattern=PATT)
    cv.fit(train_data)
    return cv


def save_vectorizer(vect):
    # Save vectorizer using pickle
    with open(f"{SPATH}/s{SID}_vectorizer.pkl", "wb") as f:
        pickle.dump(vect, file=f)


def save_model(model, filename):
    """Save both the model state and the vectorizer."""
    # Save model state
    torch.save(model.state_dict(), f"{filename}.pt")


def load_vectorizer(session_id: int) -> CountVectorizer:
    with open(f"./sessions/s{session_id}/s{session_id}_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        return vectorizer


def load_model_and_vectorizer(session_id, fname):
    """Load both the model state and the vectorizer."""

    # Load vectorizer
    vectorizer = load_vectorizer(session_id)

    # Create model with correct input size
    model = LogisticRegression(len(vectorizer.vocabulary_), 3)

    # Load model state
    model.load_state_dict(
        torch.load(f"./sessions/s{session_id}/models/{fname}.pt", weights_only=True)
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
    temp, test = train_test_split(df, test_size=test_ratio, random_state=SRST)
    train, valid = train_test_split(
        temp, test_size=valid_ratio / (train_ratio + valid_ratio), random_state=SRST
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
    plt.savefig(f"{SPATH}/plots/loss_plot.png")
    plt.close()


def main():
    # Load the raw data
    if SAMPLE_SIZE:
        df = pd.read_csv(DATA_PATH)
        df = df.sample(SAMPLE_SIZE, random_state=SRST)
        train_df, valid_df, test_df = split_data(df, (0.6, 0.2, 0.2))
    else:
        train_df, valid_df, test_df = load_data()

    print(f"Using {DEVICE} device")

    if MODE == "train":
        # preprocess it with CountVectorizer
        cv = create_vectorizer(train_df.iloc[:, 0])

        # Same vectorizer along the sessions so save it once
        os.makedirs(f"{SPATH}/models")
        os.makedirs(f"{SPATH}/plots")
        save_vectorizer(cv)

        with open(f"{SPATH}/s{SID}_info.log", "a") as f:
            f.write(
                f"Session Number: {SID}. Samples: {SAMPLE_SIZE}. Random State: {SRST}. Total Epochs: {EPOCHS}\n"
            )
            f.write(
                f"Vocabulary Size: {len(cv.vocabulary_)}. Minimum Word Frequency: {MIN_WORD_FREQ}. Pattern: "
            )
            f.write("'" + PATT + "'\n")

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

        # Evaluate it with the test data
    elif MODE == "eval":
        """
        Important: Ensure the evaluation uses the same data as the training.
        Check the '.log' file in the saved session directory and verify these parameters match:

        SAMPLE_SIZE: number of samples
        SRT: Random state for data sampling and splitting.
        SID: For correct CountVectorizer and model initialization
        """
        fname = "s2e6499"
        file_eval = open(f"{SPATH}/{fname}_eval.log", "w")
        with torch.inference_mode():
            model, cv_loaded = load_model_and_vectorizer(SID, fname)
            x_train, y_train = df2tensors(train_df, cv_loaded)
            x_valid, y_valid = df2tensors(valid_df, cv_loaded)
            x_test, y_test = df2tensors(test_df, cv_loaded)

            print(f"********** {fname.upper()} Performance *******", file=file_eval)
            print("For Train dataset:", file=file_eval)

            train_pred = model.pred(x_train)
            print(
                classification_report(y_train, train_pred, zero_division=0),
                file=file_eval,
            )
            # print(confusion_matrix(y_train, train_pred))

            print(
                f"Mean F1_Score: {f1_score(y_train, train_pred, average='macro')}",
                file=file_eval,
            )
            print("For Validation dataset:", file=file_eval)
            val_pred = model.pred(x_valid)
            print(
                classification_report(y_valid, val_pred, zero_division=0),
                file=file_eval,
            )
            print(
                f"Mean F1_Score: {f1_score(y_valid,val_pred, average='macro')}",
                file=file_eval,
            )
            print("For Test dataset:", file=file_eval)
            test_pred = model.pred(x_test)
            print(
                classification_report(y_test, test_pred, zero_division=0),
                file=file_eval,
            )
            print(
                f"Mean F1_Score: {f1_score(y_test,test_pred, average='macro')}",
                file=file_eval,
            )
            file_eval.close()


if __name__ == "__main__":
    main()
