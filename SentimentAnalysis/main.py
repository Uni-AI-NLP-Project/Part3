import os
import torch
import pandas as pd
import _pickle as pickle
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from copy import deepcopy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "train"
DATA_PATH = "./data/twitter.csv"

BS = 5000

N_CLASSES = 3
EPSILON = 1e-7
# Vectorizer Parameters
PATT = r"\S+"
MIN_WORD_FREQ = 3

SAVE_BEST = True
FIT_DEBUG = True
LR = 0.1
MAX_EPOCHS = 100
EPOCHS_PER_VALIDATION = 10
EPOCHS_PER_SAVE = 200

os.makedirs(f"models", exist_ok=True)
os.makedirs(f"plots", exist_ok=True)

accuray = torchmetrics.Accuracy(
    task="multiclass", num_classes=N_CLASSES, average="micro"
)
precision = torchmetrics.Precision(
    task="multiclass", num_classes=N_CLASSES, average="macro"
)
recall = torchmetrics.Recall(task="multiclass", num_classes=N_CLASSES, average="macro")
f1_score = torchmetrics.F1Score(
    task="multiclass", num_classes=N_CLASSES, average="macro"
)


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
    return model, torch.optim.Adam(model.parameters(), lr)


def reset_all_metrics():
    accuray.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()


def compute_all_metrics():
    return accuray.compute(), precision.compute(), recall.compute(), f1_score.compute()


def fit(
    epochs,
    model: LogisticRegression,
    loss_func,
    opt,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> tuple[list, list]:
    train_losses, valid_losses = [], []

    if SAVE_BEST:
        min_val_loss = float("inf")
        best_epoch = 0

    num_train_batches = len(train_dl)
    num_valid_batches = len(valid_dl)

    for epoch in range(epochs):
        validate = True if epoch % EPOCHS_PER_SAVE == 0 else False

        if validate:
            reset_all_metrics()
            train_cum_loss = 0

        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            train_logits = model(xb)
            train_batch_loss = loss_func(train_logits, yb)

            train_batch_loss.backward()
            opt.step()
            opt.zero_grad()

            if validate:
                model.eval()
                with torch.inference_mode():
                    train_cum_loss += train_batch_loss.item()
                    accuray.update(yb, train_logits)
                    precision.update(yb, train_logits)
                    recall.update(yb, train_logits)
                    f1_score.update(yb, train_logits)

        if validate:
            model.eval()
            with torch.inference_mode():
                train_acc, train_prec, train_rec, train_f1 = compute_all_metrics()
                train_loss = train_cum_loss / num_train_batches
                train_cum_loss = 0

                # Compute metrics on validation
                reset_all_metrics()
                valid_cum_loss = 0
                for xb, yb in valid_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    valid_logits = model(xb)
                    valid_batch_loss = loss_func(valid_logits, yb)

                    valid_cum_loss += valid_batch_loss.item()
                    accuray.update(yb, valid_logits)
                    precision.update(yb, valid_logits)
                    recall.update(yb, valid_logits)
                    f1_score.update(yb, valid_logits)

                valid_acc, valid_prec, valid_rec, valid_f1 = compute_all_metrics()
                valid_loss = valid_cum_loss / num_valid_batches
                valid_cum_loss = 0

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.5f}, Val Loss = {valid_loss:.5f} | "
                    f"Train Acc = {train_acc:.5f}, Val Acc = {valid_acc:.5f} | Train F1 = {train_f1:.5f}, "
                    f"Val F1 = {valid_f1:.5f}"
                )

    return train_losses, valid_losses


def con_mat(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate confusion matrix for multi-class classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        torch.Tensor: N_CLASSES x N_CLASSES confusion matrix
    """
    cm = torch.zeros((N_CLASSES, N_CLASSES), device=DEVICE)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            cm[i, j] = ((y_true == i) & (y_pred == j)).sum()
    return cm


def calculate_metrics(confusion_matrix: torch.Tensor) -> dict:
    """
    Calculate precision, recall, and F1-score for each class.

    Args:
        confusion_matrix: N_CLASSES x N_CLASSES confusion matrix

    Returns:
        dict: Dictionary containing precision, recall, and F1-score for each class,
              plus macro-averaged F1 score
    """
    metrics = {}
    n_classes = confusion_matrix.shape[0]

    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)
        f1 = 2 * precision * recall / (precision + recall + EPSILON)

        metrics[f"class_{i}"] = {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

    metrics["macro_f1"] = (
        sum(metrics[f"class_{i}"]["f1"] for i in range(n_classes)) / n_classes
    )
    return metrics


def create_vectorizer(train_data) -> CountVectorizer:
    cv = CountVectorizer(min_df=MIN_WORD_FREQ, binary=True, token_pattern=PATT)
    cv.fit(train_data)
    return cv


def save_vectorizer(vect):
    # Save vectorizer using pickle
    with open(f"vectorizer.pkl", "wb") as f:
        pickle.dump(vect, file=f)


def save_model(model, filename):
    """Save both the model state and the vectorizer."""
    # Save model state
    torch.save(model.state_dict(), f"{filename}.pt")


def load_vectorizer(path) -> CountVectorizer:
    with open(os.path.join(path), "rb") as f:
        vectorizer = pickle.load(f)
        return vectorizer


def load_model_and_vectorizer(model_fname, vect_fname):
    """Load both the model state and the vectorizer."""

    # Load vectorizer
    vectorizer = load_vectorizer(vect_fname)

    # Create model with correct input size
    model = LogisticRegression(len(vectorizer.vocabulary_), 3)

    # Load model state
    model.load_state_dict(torch.load(f"./models/{model_fname}.pt", weights_only=True))

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
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
    return x, y


# def to_dataloader(x, y):
#     ds = TensorDataset(x, y)
#     dl = DataLoader(dataset=ds,
#                     pin_memory=True)


def save_plots(train_ls, valid_ls):
    plt.figure(figsize=(16, 9))
    plt.plot(train_ls, label="Train Loss", color="blue")
    plt.plot(valid_ls, label="Valid Loss", color="red")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"plots/loss_plot.png")
    plt.close()


def main():
    # Load the cleaned data

    train_df, valid_df, test_df = load_data()

    print(f"Using {DEVICE} device")

    if MODE == "train":
        # preprocess it with CountVectorizer
        cv = create_vectorizer(train_df.iloc[:, 0])

        # Same vectorizer along the sessions so save it once

        save_vectorizer(cv)

        with open(f"info.log", "a") as f:
            f.write(
                f"Vocabulary Size: {len(cv.vocabulary_)}. Minimum Word Frequency: {MIN_WORD_FREQ}. Pattern: "
            )
            f.write("'" + PATT + "'\n")

        x_train, y_train = df2tensors(train_df, cv)
        x_valid, y_valid = df2tensors(valid_df, cv)

        # Wrap in DataLoaeders
        train_loader = DataLoader(
            dataset=TensorDataset(x_train, y_train),
            batch_size=BS,
            shuffle=True,
            pin_memory=True,
        )

        valid_loader = DataLoader(
            dataset=TensorDataset(x_train, y_train),
            batch_size=2 * BS,
            shuffle=False,
            pin_memory=True,
        )
        # Create Model
        num_features = len(cv.vocabulary_)

        model, opt = get_model(num_features, N_CLASSES, lr=LR)
        model = model.to(DEVICE)

        # Train the model(s)
        criteria = nn.CrossEntropyLoss()

        train_losses, valid_losses = fit(
            MAX_EPOCHS, model, criteria, opt, train_loader, valid_loader
        )
        # Save plots
        save_plots(train_losses, valid_losses)
        save_model(model, f"{MAX_EPOCHS}_epochs")
        # Evaluate it with the test data
    elif MODE == "eval":
        """
        Important: Ensure the evaluation uses the same data as the training.
        Check the '.log' file in the saved session directory and verify these parameters match:

        SAMPLE_SIZE: number of samples
        SRT: Random state for data sampling and splitting.
        SID: For correct CountVectorizer and model initialization
        """
        model_fname = "s2e6499"
        vect_fname = ""
        file_eval = open(f"{model_fname}_eval.log", "w")
        with torch.inference_mode():
            model, cv_loaded = load_model_and_vectorizer(model_fname, vect_fname)
            model.eval()
            x_train, y_train = df2tensors(
                train_df,
                cv_loaded,
            )
            x_valid, y_valid = df2tensors(valid_df, cv_loaded)
            x_test, y_test = df2tensors(test_df, cv_loaded)

            print(
                f"********** {model_fname.upper()} Performance *******", file=file_eval
            )
            print("For Train dataset:", file=file_eval)

            train_pred = model.pred(x_train)
            train_cm = con_mat(y_train, train_pred)
            print(
                calculate_metrics(train_cm),
                file=file_eval,
            )

            print("For Validation dataset:", file=file_eval)
            val_pred = model.pred(x_valid)
            val_cm = con_mat(y_train, val_pred)
            print(calculate_metrics(val_cm), file=file_eval)

            print("For Test dataset:", file=file_eval)
            test_pred = model.pred(x_test)
            test_cm = con_mat(y_test, test_pred)
            print(calculate_metrics(test_cm), file=file_eval)
            file_eval.close()


if __name__ == "__main__":
    main()
