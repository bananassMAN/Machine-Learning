import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


DATA_PATH = Path(__file__).resolve().with_name("placement_predict_50k.csv")


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(X, y, w, b):
    z = X @ w + b
    p = sigmoid(z)
    eps = 1e-8
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()


def accuracy(X, y, w, b):
    p = sigmoid(X @ w + b)
    pred = (p >= 0.5).astype(float)
    return np.mean(pred == y)


def batch_gradient_descent(X, y, lr=0.05, epochs=200):
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0
    losses = []
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        dw = (1 / n) * (X.T @ (p - y))
        db = (1 / n) * np.sum(p - y)
        w -= lr * dw
        b -= lr * db
        losses.append(compute_loss(X, y, w, b))
    return w, b, losses


def mini_batch_gradient_descent(X, y, lr=0.05, epochs=200, batch_size=32):
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0
    losses = []
    for _ in range(epochs):
        idx = np.random.permutation(n)
        Xs = X[idx]
        ys = y[idx]
        for start in range(0, n, batch_size):
            end = start + batch_size
            Xb = Xs[start:end]
            yb = ys[start:end]
            p = sigmoid(Xb @ w + b)
            dw = (1 / len(Xb)) * (Xb.T @ (p - yb))
            db = (1 / len(Xb)) * np.sum(p - yb)
            w -= lr * dw
            b -= lr * db
        losses.append(compute_loss(X, y, w, b))
    return w, b, losses


def load_or_create_data(path):
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded dataset from: {path}")
        return df

    print(f"Dataset not found at {path}. Creating a demo dataset instead.")
    rng = np.random.default_rng(42)
    n = 50000

    CGPA = rng.uniform(5.0, 10.0, size=n)
    Internships = rng.integers(0, 6, size=n)
    Projects = rng.integers(0, 8, size=n)
    AptitudeScore = rng.uniform(40, 100, size=n)
    SoftSkillsRating = rng.uniform(1, 10, size=n)
    Backlogs = rng.integers(0, 4, size=n)
    PlacementTraining = rng.choice(["Yes", "No"], size=n, p=[0.55, 0.45])

    logit = (
        -7.0
        + 1.2 * CGPA
        + 0.35 * Internships
        + 0.30 * Projects
        + 0.06 * AptitudeScore
        + 0.45 * SoftSkillsRating
        - 0.60 * Backlogs
        + 0.90 * (PlacementTraining == "Yes")
    )
    prob = sigmoid(logit)
    PlacementStatus = np.where(rng.random(n) < prob, "Placed", "Not Placed")

    df = pd.DataFrame({
        "CGPA": CGPA,
        "Internships": Internships,
        "Projects": Projects,
        "AptitudeScore": AptitudeScore,
        "SoftSkillsRating": SoftSkillsRating,
        "Backlogs": Backlogs,
        "PlacementTraining": PlacementTraining,
        "PlacementStatus": PlacementStatus,
    })

    df.to_csv(path, index=False)
    print(f"Saved demo dataset to: {path}")
    return df


def main():
    df = load_or_create_data(DATA_PATH)

    df["PlacementTraining"] = (df["PlacementTraining"] == "Yes").astype(float)

    features = [
        "CGPA",
        "Internships",
        "Projects",
        "AptitudeScore",
        "SoftSkillsRating",
        "Backlogs",
        "PlacementTraining",
    ]

    X = df[features].to_numpy(dtype=float)
    y = (df["PlacementStatus"] == "Placed").astype(float).to_numpy(dtype=float)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    w_batch, b_batch, loss_batch = batch_gradient_descent(X, y, lr=0.05, epochs=200)
    w_mini, b_mini, loss_mini = mini_batch_gradient_descent(X, y, lr=0.05, epochs=200, batch_size=32)

    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    linear_mse = mean_squared_error(y, y_pred_linear)

    batch_acc = accuracy(X, y, w_batch, b_batch)
    mini_acc = accuracy(X, y, w_mini, b_mini)

    print("Batch accuracy:", batch_acc)
    print("Mini-batch accuracy:", mini_acc)
    print("Linear regression MSE:", linear_mse)
    print("Batch final loss:", loss_batch[-1])
    print("Mini-batch final loss:", loss_mini[-1])

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(loss_batch, label="Batch GD", color="blue", linewidth=2)
    axes[0, 0].set_title("Batch Gradient Descent Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(loss_mini, label="Mini-batch GD", color="red", linewidth=2)
    axes[0, 1].set_title("Mini-batch Gradient Descent Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(y, y_pred_linear, alpha=0.25, color="green")
    axes[1, 0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Linear Regression: Actual vs Predicted")
    axes[1, 0].set_xlabel("Actual")
    axes[1, 0].set_ylabel("Predicted")
    axes[1, 0].grid(True, alpha=0.3)

    final_losses = {
        "Batch GD": loss_batch[-1],
        "Mini-batch GD": loss_mini[-1],
        "Linear Regression": linear_mse,
    }

    axes[1, 1].bar(final_losses.keys(), final_losses.values(), color=["blue", "red", "green"])
    axes[1, 1].set_title("Final Loss Comparison")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).resolve().with_name("placement_model_comparison_plots.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved comparison plots to: {out_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_batch, label="Batch Gradient Descent", color="blue", linewidth=2)
    plt.plot(loss_mini, label="Mini-batch Gradient Descent", color="red", linewidth=2)
    plt.title("Batch vs Mini-batch Gradient Descent")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().with_name("batch_vs_minibatch_loss.png"), dpi=200)
    print(f"Saved combined loss plot to: {Path(__file__).resolve().with_name('batch_vs_minibatch_loss.png')}")

    plt.figure(figsize=(8, 5))
    accuracy_values = {
        "Batch GD": batch_acc,
        "Mini-batch GD": mini_acc,
        "Linear Regression": 1.0 - linear_mse,
    }
    plt.bar(accuracy_values.keys(), accuracy_values.values(), color=["blue", "red", "green"])
    plt.title("Accuracy / Fit Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().with_name("accuracy_comparison.png"), dpi=200)
    print(f"Saved accuracy comparison plot to: {Path(__file__).resolve().with_name('accuracy_comparison.png')}")


if __name__ == "__main__":
    main()
