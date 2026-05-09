import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from skimage.feature import hog


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
        self.n_features = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        n_samples = len(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts) + self.alpha * self.n_features
            self.feature_probs[c] = (feature_counts + self.alpha) / total_count

    def _compute_log_likelihood(self, X, c):
        return np.sum(X * np.log(self.feature_probs[c] + 1e-10), axis=1)

    def predict(self, X):
        log_posteriors = np.zeros((len(X), len(self.classes)))

        for i, c in enumerate(self.classes):
            log_prior = np.log(self.class_priors[c] + 1e-10)
            log_likelihood = self._compute_log_likelihood(X, c)
            log_posteriors[:, i] = log_prior + log_likelihood

        return self.classes[np.argmax(log_posteriors, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def predict_proba(self, X):
        log_posteriors = np.zeros((len(X), len(self.classes)))
        for i, c in enumerate(self.classes):
            log_prior = np.log(self.class_priors[c] + 1e-10)
            log_likelihood = self._compute_log_likelihood(X, c)
            log_posteriors[:, i] = log_prior + log_likelihood
        exp_log_post = np.exp(log_posteriors - log_posteriors.max(axis=1, keepdims=True))
        return exp_log_post / exp_log_post.sum(axis=1, keepdims=True)

    def save(self, filepath):
        model_data = {
            "classes": self.classes.tolist(),
            "class_priors": {str(k): v for k, v in self.class_priors.items()},
            "feature_probs": {str(k): v.tolist() for k, v in self.feature_probs.items()},
            "n_features": self.n_features,
            "alpha": self.alpha
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)


def extract_hog(X, cell_size=3):
    X_reshaped = X.reshape(-1, 28, 28)
    features = []
    for img in X_reshaped:
        hog_feat = hog(img, orientations=9, pixels_per_cell=(cell_size, cell_size),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_feat)
    return np.array(features)


def compute_confusion_matrix(y_true, y_pred, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1
    return cm


def print_confusion_matrix(cm, classes):
    print("\nConfusion Matrix:")
    header = "Pred:   " + "".join(f"{c:>4}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"True {i}:" + "".join(f"{val:>5}" for val in row)
        print(row_str)


def plot_confusion_matrix(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Digit', fontsize=12)
    ax.set_ylabel('True Digit', fontsize=12)
    ax.set_title('Confusion Matrix - MultinomialNB with HOG 3x3', fontsize=14)
    plt.colorbar(im, ax=ax)
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix heatmap saved to: {save_path}")


def compute_metrics(y_true, y_pred, classes):
    precision, recall, f1 = [], [], []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f)

    accuracy = np.mean(y_pred == y_true)
    macro_f1 = np.mean(f1)

    return accuracy, macro_f1, precision, recall, f1


def normalize(X):
    return (X - X.min()) / (X.max() - X.min() + 1e-10)


def plot_class_priors(class_priors, classes, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    values = [class_priors[c] for c in classes]
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, values, color=colors, edgecolor='black')
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Prior Probability', fontsize=12)
    ax.set_title('Class Priors - MultinomialNB', fontsize=14)
    ax.set_ylim(0, max(values) * 1.1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Class priors chart saved to: {save_path}")


def plot_feature_probs_heatmap(feature_probs, classes, save_path, top_n=100):
    probs_matrix = np.array([feature_probs[c] for c in classes])
    top_indices = np.argsort(np.var(probs_matrix, axis=0))[-top_n:]
    probs_subset = probs_matrix[:, top_indices]
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(probs_subset, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Top Discriminative HOG Features (sorted by variance)', fontsize=12)
    ax.set_ylabel('Digit Class', fontsize=12)
    ax.set_title(f'Feature Probability Heatmap - Top {top_n} Features per Class', fontsize=14)
    plt.colorbar(im, ax=ax, label='P(feature|class)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature probabilities heatmap saved to: {save_path}")


def plot_feature_importance_overlay(feature_probs, classes, save_path, top_n=15):
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    axes = axes.flatten()
    for idx, digit in enumerate(classes):
        ax = axes[idx]
        probs = feature_probs[digit]
        top_indices = np.argsort(probs)[-top_n:]
        top_values = probs[top_indices]
        ax.barh(range(top_n), top_values, color='steelblue', edgecolor='black')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f"F{i}" for i in top_indices], fontsize=8)
        ax.set_xlabel('Feature Probability', fontsize=10)
        ax.set_title(f'Digit {digit}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle('Top Discriminative HOG Features per Digit Class', fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance chart saved to: {save_path}")


def plot_alpha_sensitivity(X_train, y_train, X_val, y_val, classes, save_path, alphas=None):
    if alphas is None:
        alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    train_accs, val_accs = [], []
    print("\nAlpha sensitivity analysis...")
    for alpha in alphas:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        train_accs.append(model.score(X_train, y_train))
        val_accs.append(model.score(X_val, y_val))
        print(f"  alpha={alpha:.4f} -> train={train_accs[-1]*100:.2f}%, val={val_accs[-1]*100:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, train_accs, 'o-', label='Train Accuracy', color='blue', linewidth=2)
    ax.plot(alphas, val_accs, 's-', label='Validation Accuracy', color='orange', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (Laplace Smoothing)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Alpha Sensitivity Analysis - MultinomialNB', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    best_idx = np.argmax(val_accs)
    ax.axvline(alphas[best_idx], color='green', linestyle='--', label=f'Best alpha={alphas[best_idx]}')
    ax.legend(fontsize=11)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Alpha sensitivity curve saved to: {save_path}")
    return alphas[best_idx]


def plot_log_posterior_distribution(model, X_sample, y_sample, classes, save_path, n_samples=5):
    fig, axes = plt.subplots(1, n_samples, figsize=(20, 4))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]
        X_single = X_sample[i:i+1]
        log_posteriors = np.zeros(len(classes))
        for j, c in enumerate(classes):
            log_prior = np.log(model.class_priors[c] + 1e-10)
            log_likelihood = model._compute_log_likelihood(X_single, c)[0]
            log_posteriors[j] = log_prior + log_likelihood

        colors = ['green' if c == y_sample[i] else 'red' for c in classes]
        bars = ax.bar(range(len(classes)), log_posteriors, color=colors, edgecolor='black')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_xlabel('Digit Class', fontsize=10)
        ax.set_ylabel('Log Posterior', fontsize=10)
        true_label = y_sample[i]
        pred_label = classes[np.argmax(log_posteriors)]
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', fontsize=10)
        confidence = np.max(log_posteriors) - np.sort(log_posteriors)[-2]
        ax.text(0.05, 0.95, f'Conf: {confidence:.2f}', transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Log-Posterior Distribution per Sample (Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Log-posterior distribution saved to: {save_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    print("=" * 70)
    print("Final Model: MultinomialNB + HOG 3x3")
    print("=" * 70)

    print("\nLoading MNIST data...")
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 784).astype(float) / 255.0
    X_test = X_test.reshape(-1, 784).astype(float) / 255.0

    print("\nSplitting training data (80% train, 20% validation)...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    print("\nExtracting HOG features (cell_size=3)...")
    start = time.time()
    X_train_hog = extract_hog(X_train, cell_size=3)
    X_val_hog = extract_hog(X_val, cell_size=3)
    X_test_hog = extract_hog(X_test, cell_size=3)
    print(f"Feature dimension: {X_train_hog.shape[1]}")
    print(f"Extraction time: {time.time() - start:.1f}s")

    print("\nNormalizing features...")
    X_train_norm = normalize(X_train_hog)
    X_val_norm = normalize(X_val_hog)
    X_test_norm = normalize(X_test_hog)

    classes = np.arange(10)

    print("\n[1] Class Prior Bar Chart...")
    print("-" * 50)
    for c in classes:
        print(f"  Class {c}: prior = {len(X_train[y_train == c]) / len(X_train):.4f}")
    model = MultinomialNB(alpha=0.01)
    model.fit(X_train_norm, y_train)
    plot_class_priors(model.class_priors, classes, "results/class_priors.png")

    print("\n[2] Feature Probability Heatmap...")
    print("-" * 50)
    plot_feature_probs_heatmap(model.feature_probs, classes, "results/feature_probs_heatmap.png")

    print("\n[3] Feature Importance Overlay...")
    print("-" * 50)
    plot_feature_importance_overlay(model.feature_probs, classes, "results/feature_importance_overlay.png")

    print("\n[4] Alpha Sensitivity Analysis...")
    print("-" * 50)
    best_alpha = plot_alpha_sensitivity(X_train_norm, y_train, X_val_norm, y_val, classes, "results/alpha_sensitivity_curve.png")
    print(f"Best alpha found: {best_alpha}")

    print(f"\nRetraining with best alpha={best_alpha}...")
    model = MultinomialNB(alpha=best_alpha)
    model.fit(X_train_norm, y_train)

    print("\n[5] Log-Posterior Distribution...")
    print("-" * 50)
    sample_indices = np.random.choice(len(X_test_norm), 5, replace=False)
    X_sample = X_test_norm[sample_indices]
    y_sample = y_test[sample_indices]
    plot_log_posterior_distribution(model, X_sample, y_sample, classes, "results/log_posterior_sample.png")

    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_norm)

    accuracy, macro_f1, precision, recall, f1 = compute_metrics(y_test, y_pred, classes)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (on Test Set)")
    print("=" * 70)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Best Alpha: {best_alpha}")

    print("\nPer-Class Metrics:")
    print("-" * 55)
    print(f"{'Digit':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 55)
    for i in range(10):
        print(f"{i:<6} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    print("-" * 55)

    cm = compute_confusion_matrix(y_test, y_pred, classes)
    print_confusion_matrix(cm, classes)
    plot_confusion_matrix(cm, classes, "results/final_confusion_matrix.png")

    model.save("final_multinomialNB_model.json")
    print(f"\nModel saved to: final_multinomialNB_model.json")

    print("\n" + "=" * 70)
    print("VISUALIZATIONS GENERATED")
    print("=" * 70)
    print("  1. class_priors.png              - Class prior bar chart")
    print("  2. feature_probs_heatmap.png    - Feature probability heatmap")
    print("  3. feature_importance_overlay.png - Feature importance overlay")
    print("  4. alpha_sensitivity_curve.png   - Alpha sensitivity analysis")
    print("  5. log_posterior_sample.png     - Log-posterior distribution")
    print("  6. final_confusion_matrix.png   - Confusion matrix heatmap")
    print("=" * 70)


if __name__ == "__main__":
    main()