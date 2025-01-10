import os
import shutil
import torch
import pandas as pd
import numpy as np
import _pickle as pickle
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "train"

KAGGLE_DATA_PATH = "/kaggle/input/twitter-ds"
KAGGLE_OUTPUT_PATH = "/kaggle/working/"

LOCAL_DATA_PATH = "./small_data"
LOCAL_OUTPUT_PATH = "./s1"

DATA_PATH = LOCAL_DATA_PATH
OUTPUT_PATH = LOCAL_OUTPUT_PATH


BS = 40000

N_CLASSES = 3
EPSILON = 1e-7
# Vectorizer Parameters
PATT = r"\S+"
MIN_WORD_FREQ = 3

SAVE_BEST = True
LR = 0.1
MAX_EPOCHS = 50
EPOCHS_PER_VALIDATION = 5
SCHEDULER_PATIENCE = 5
EPOCHS_PER_CHPNT = 10
PATIENCE_LIMIT = None

DEBUG = True


accuray = torchmetrics.Accuracy(
    task="multiclass", num_classes=N_CLASSES, average="micro"
).to(DEVICE)
precision = torchmetrics.Precision(
    task="multiclass", num_classes=N_CLASSES, average="macro"
).to(DEVICE)
recall = torchmetrics.Recall(
    task="multiclass", num_classes=N_CLASSES, average="macro"
).to(DEVICE)
f1_score = torchmetrics.F1Score(
    task="multiclass", num_classes=N_CLASSES, average="macro"
).to(DEVICE)


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


class MetricsLog:
    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }

    def add(self, loss, acc, pre, rec, f1):
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(acc)
        self.metrics["precision"].append(pre)
        self.metrics["recall"].append(rec)
        self.metrics["f1_score"].append(f1)


def get_model(n_features, n_classes, lr: float = 0.001):
    """
    Creates a logistic regression model with the specified input and output dimensions
    and initializes the Adam optimizer with a given learning rate.

    Args:
        n_features (int): The number of features in the input data.
        n_classes (int): The number of target classes for classification.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        tuple:
            - LogisticRegression: The initialized model.
            - torch.optim.Adam: Optimizer for training the model.
    """
    model = LogisticRegression(n_features, n_classes)
    return model, torch.optim.Adam(model.parameters(), lr)


def update_metrics(pred, true):
    """
    Updates the accuracy, precision, recall, and F1 metrics using predicted and actual labels.

    Args:
        pred (torch.Tensor): Predicted labels for the batch.
        true (torch.Tensor): True labels for the batch.

    Note:
        Metrics are aggregated across batches and reset before validation.
    """
    accuray.update(pred, true)
    precision.update(pred, true)
    recall.update(pred, true)
    f1_score.update(pred, true)


def reset_all_metrics():
    """
    Resets the stored values of accuracy, precision, recall, and F1-score.
    Should be called before computing metrics for a new dataset or epoch.
    """

    accuray.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()


def compute_all_metrics():
    """
    Computes aggregated accuracy, precision, recall, and F1-score.

    Returns:
        tuple:
            - float: Accuracy score.
            - float: Precision score (macro-averaged).
            - float: Recall score (macro-averaged).
            - float: F1-score (macro-averaged).
    """
    return (
        accuray.compute().item(),
        precision.compute().item(),
        recall.compute().item(),
        f1_score.compute().item(),
    )


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    """
    Implements the training loop for a logistic regression model, with validation
    at specified intervals and support for learning rate scheduling and early stopping.

    Args:
        epochs (int): Total number of epochs to train the model.
        model (LogisticRegression): The logistic regression model to be trained.
        loss_func (nn.Module): Loss function (e.g., CrossEntropyLoss).
        opt (torch.optim.Optimizer): Optimizer for gradient descent.
        train_dl (DataLoader): DataLoader for training data.
        valid_dl (DataLoader): DataLoader for validation data.

    Returns:
        tuple:
            - MetricsLog: Training metrics logged over epochs.
            - MetricsLog: Validation metrics logged over epochs.

    Note:
        Best-performing model based on validation loss is saved during training.
    """

    train_metrics, valid_metrics = MetricsLog(), MetricsLog()
    log_fname = open(os.path.join(OUTPUT_PATH, "training_stats.log"), "w")
    num_train_batches = len(train_dl)
    num_valid_batches = len(valid_dl)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    patience_counter = 0

    if SAVE_BEST:
        min_valid_loss = float("inf")
        min_valid_f1 = float("inf")

    for epoch in range(epochs + 1):
        validate = True if epoch % EPOCHS_PER_VALIDATION == 0 else False

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
                    update_metrics(train_logits, yb)

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
                    update_metrics(valid_logits, yb)

                valid_acc, valid_prec, valid_rec, valid_f1 = compute_all_metrics()
                valid_loss = valid_cum_loss / num_valid_batches
                scheduler.step(valid_loss)
                valid_cum_loss = 0

                train_metrics.add(
                    train_loss, train_acc, train_prec, train_rec, train_f1
                )
                valid_metrics.add(
                    valid_loss, valid_acc, valid_prec, valid_rec, valid_f1
                )

                stats = (
                    f"Epoch {epoch}: Train Loss = {train_loss:.5f}, Val Loss = {valid_loss:.5f} | "
                    f"Train Acc = {train_acc:.5f}, Val Acc = {valid_acc:.5f} | Train F1 = {train_f1:.5f}, "
                    f"Val F1 = {valid_f1:.5f}"
                )
                print(stats)
                log_fname.write(f"{stats}\n")

            if SAVE_BEST and valid_loss < min_valid_loss:
                last_f1 = valid_metrics.metrics["f1_score"][-1]
                if valid_loss < min_valid_loss:
                    best_model_sd = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    min_valid_loss = valid_loss
                if last_f1 < min_valid_f1:
                    best_f1 = {k: v.cpu() for k, v in model.state_dict().items()}
                    min_valid_f1 = last_f1

            if epoch > 0 and PATIENCE_LIMIT is not None:
                # update counts
                prev_valid_loss = valid_metrics.metrics["loss"][-2]
                patience_counter = (
                    patience_counter + 1 if valid_loss >= prev_valid_loss else 0
                )
                if DEBUG:
                    print(
                        f"Pateience_counter: {patience_counter}, val_loss: {valid_loss}, prev_loss: {prev_valid_loss}"
                    )
                if patience_counter >= PATIENCE_LIMIT:
                    print(f"Early stopping triggered ({PATIENCE_LIMIT} bad epochs)")
                    break
            if DEBUG:
                print(f"get_lr: {scheduler.get_last_lr()}")
                acc_m = train_metrics.metrics["accuracy"]
                if len(acc_m) > 0:
                    print(f"type of saved stats in metric log: {type(acc_m[0])}")
                print(type(train_metrics.metrics["accuracy"]))

        if epoch % EPOCHS_PER_CHPNT == 0:
            save_model(model, f"epoch_{epoch}_checkpoint")

    if SAVE_BEST:
        torch.save(
            best_model_sd,
            os.path.join(
                OUTPUT_PATH, "models", f"best_loss_model_epoch{best_epoch}.pt"
            ),
        )
        torch.save(
            best_model_sd,
            os.path.join(OUTPUT_PATH, "models", f"best_f1_score:{min_valid_f1:.4f}.pt"),
        )

    log_fname.close()

    return train_metrics, valid_metrics


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
    """
    Initialize and fit a CountVectorizer on the training data.

    Args:
        train_data (iterable): Training data containing text samples.

    Returns:
        CountVectorizer: Fitted vectorizer instance.
    """

    cv = CountVectorizer(min_df=MIN_WORD_FREQ, binary=True, token_pattern=PATT)
    cv.fit(train_data)
    return cv


def save_vectorizer(vect):
    """
    Save a fitted CountVectorizer to a file.

    Args:
        vect (CountVectorizer): Fitted vectorizer to save.
    """
    # Save vectorizer using pickle
    with open(os.path.join(OUTPUT_PATH, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vect, file=f)


def save_model(model, filename):
    """
    Save a PyTorch model to a file.

    Args:
        model (nn.Module): The model to save.
        filename (str): Name of the file for saving the model.
    """

    # Save model state
    torch.save(
        model.state_dict(), os.path.join(OUTPUT_PATH, "models", f"{filename}.pt")
    )


def load_vectorizer(fname) -> CountVectorizer:
    """
    Load a saved CountVectorizer from a file.

    Args:
        fname (str): Name of the file containing the vectorizer.

    Returns:
        CountVectorizer: Loaded vectorizer instance.
    """

    with open(os.path.join(OUTPUT_PATH, f"{fname}.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
        return vectorizer


def load_model_and_vectorizer(model_fname, vect_fname):
    """
    Load a saved model and vectorizer from files.

    Args:
        model_fname (str): File name of the model.
        vect_fname (str): File name of the vectorizer.

    Returns:
        tuple: Loaded model and vectorizer.
    """

    # Load vectorizer
    vectorizer = load_vectorizer(vect_fname)

    # Create model with correct input size
    model = LogisticRegression(len(vectorizer.vocabulary_), N_CLASSES)

    # Load model state
    model.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_PATH, "models", f"{model_fname}.pt"), weights_only=True
        )
    )

    return model, vectorizer


def load_data():
    """
    Load the training, validation, and test datasets.

    Returns:
        tuple: DataFrames for training, validation, and test datasets.
    """

    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    valid = pd.read_csv(os.path.join(DATA_PATH, "valid.csv"))
    test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    return train, valid, test


def df2tensors(df: pd.DataFrame, cv: CountVectorizer):
    """
    Convert a DataFrame to tensors using a fitted CountVectorizer.

    Args:
        df (pd.DataFrame): DataFrame containing text and label columns.
        cv (CountVectorizer): Fitted vectorizer for text transformation.

    Returns:
        tuple: Feature tensor (X) and label tensor (Y).
    """

    x = df.iloc[:, 0]
    x = cv.transform(x).toarray()  # type: ignore
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
    return x, y


def save_plots(train_stats: MetricsLog, valid_stats: MetricsLog):
    """
    Generate and save training and validation metric plots.

    Args:
        train_stats (MetricsLog): Metrics log for training.
        valid_stats (MetricsLog): Metrics log for validation.
    """

    x = np.arange(0, len(train_stats.metrics["loss"])) * EPOCHS_PER_VALIDATION

    for metric_name, train_values in train_stats.metrics.items():
        validation_values = valid_stats.metrics[metric_name]

        plt.figure(figsize=(16, 9))
        plt.plot(x, train_values, label="Train Loss", color="blue")
        plt.plot(x, validation_values, label="Valid Loss", color="red")
        plt.title(metric_name.capitalize())
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name} During Training")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "plots", f"{metric_name}_comparison.png"))
        plt.close()


def main():
    # Load the cleaned data
    train_df, valid_df, test_df = load_data()

    print(f"Using {DEVICE} device")

    if MODE == "train":

        if OUTPUT_PATH != KAGGLE_OUTPUT_PATH and os.path.exists(OUTPUT_PATH):
            ans = input(
                "output path already exists, delete and continue with training? (y/n)? "
            )
            if ans == "y":
                shutil.rmtree(OUTPUT_PATH)
            else:
                exit()

        os.makedirs(os.path.join(OUTPUT_PATH, "models"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "plots"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "evaluations"), exist_ok=True)

        # preprocess it with CountVectorizer
        cv = create_vectorizer(train_df.iloc[:, 0])
        # Same vectorizer along the sessions so save it once
        save_vectorizer(cv)

        with open(os.path.join(OUTPUT_PATH, "info.log"), "w") as f:
            f.write(
                f"Total Epochs: {MAX_EPOCHS}, Batch Size: {BS}, Epochs_per_validation: {EPOCHS_PER_VALIDATION}\n"
            )
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
            dataset=TensorDataset(x_valid, y_valid),
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

        train_stats, valid_stats = fit(
            MAX_EPOCHS, model, criteria, opt, train_loader, valid_loader
        )

        # Save plots
        save_plots(train_stats, valid_stats)
        # Save the model after all epochs
        save_model(model, f"final_model")
        # Evaluate it with the test data
    elif MODE == "eval":
        model_fname = "best_model_epoch20"
        vect_fname = "vectorizer"
        file_eval = open(
            os.path.join(OUTPUT_PATH, "evaluations", f"{model_fname}_eval.log"), "w"
        )
        with torch.inference_mode():
            model, cv_loaded = load_model_and_vectorizer(model_fname, vect_fname)
            model = model.cpu()
            model.eval()

            x_test, y_test = df2tensors(test_df, cv_loaded)

            print(
                f"********** {model_fname.upper()} Performance *******", file=file_eval
            )

            print("For Test dataset:", file=file_eval)
            test_pred = model.pred(x_test)
            test_cm = con_mat(y_test, test_pred)
            print(test_cm, file=file_eval)
            print(calculate_metrics(test_cm), file=file_eval)
            print(
                classification_report(y_test.numpy(), test_pred.numpy()), file=file_eval
            )
            file_eval.close()

            disp = ConfusionMatrixDisplay(
                test_cm.int().numpy(),
                display_labels=["negative", "neutral", "positive"],
            )
            disp.plot(cmap="Greys")
            plt.savefig(
                os.path.join(OUTPUT_PATH, "evaluations", f"{model_fname}_cm.png"),
                dpi=300,
            )
            plt.close()


if __name__ == "__main__":
    main()
