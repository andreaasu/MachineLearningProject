import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# 1) Load and  Prepare Data

print("Loading MNIST...")

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

X_train_full = X_train_full.reshape(-1, 784) / 255.0
X_test= X_test.reshape(-1, 784) / 255.0

np.random.seed(42)
indices = np.arange(len(X_train_full))
np.random.shuffle(indices)

X_train_full = X_train_full[indices]
y_train_full = y_train_full[indices]

# train/validation split: 85% train, 15% validation
val_size = int(0.15 * len(X_train_full))

X_val = X_train_full[:val_size]
y_val = y_train_full[:val_size]

X_train = X_train_full[val_size:]
y_train = y_train_full[val_size:]

print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)


# 2) Evaluation Functions

def build_confusion_matrix(actual, predicted, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(actual)):
        matrix[int(actual[i]), int(predicted[i])] += 1

    return matrix


def evaluate_model(actual, predicted, n_classes=10):
    matrix = build_confusion_matrix(actual, predicted, n_classes)

    accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)

    precision_vals = []
    recall_vals = []
    f1_vals = []

    for digit in range(n_classes):
        TP = matrix[digit, digit]
        FP = np.sum(matrix[:, digit]) - TP
        FN = np.sum(matrix[digit, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

    return (
        accuracy,
        np.mean(precision_vals),
        np.mean(recall_vals),
        np.mean(f1_vals),
        matrix
    )


def print_results(header, actual, predicted):
    accuracy, precision, recall, f1, matrix = evaluate_model(actual, predicted)

    print("\n" + header)
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")

    return matrix


def draw_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    for row in range(10):
        for col in range(10):
            plt.text(
                col,
                row,
                matrix[row, col],
                ha="center",
                va="center",
                fontsize=7
            )

    plt.colorbar()
    plt.tight_layout()
    plt.show()


# 3) Linear SVM multiclass
class linearSVM:
    def __init__(self, C=0.1, Learning_rate=0.001, n_epochs=20, regularization="l2"):
        self.C = C
        self.Learning_rate = Learning_rate
        self.n_epochs = n_epochs
        self.regularization = regularization

    def fit(self, X, y):
        self.classes = np.arange(10)
        self.classifiers = {}

        for cls in self.classes:
            print(f"  Training class {cls} vs rest...")

            y_binary = np.where(y == cls, 1, -1)

            num_samples, num_features = X.shape
            weights = np.zeros(num_features)
            bias = 0.0

            for epoch in range(self.n_epochs):
                for idx, sample in enumerate(X):
                    score = np.dot(sample, weights) + bias
                    margin = y_binary[idx] * score

                    if margin >= 1:
                        if self.regularization == "l2":
                            weights -= self.Learning_rate * (2 / num_samples) * weights
                        else:
                            weights -= self.Learning_rate * (1 / num_samples) * np.sign(weights)

                    else:
                        if self.regularization == "l2":
                            weights -= self.Learning_rate * (
                                (2 / num_samples) * weights
                                - self.C * y_binary[idx] * sample
                            )
                        else:
                            weights -= self.Learning_rate * (
                                (1 / num_samples) * np.sign(weights)
                                - self.C * y_binary[idx] * sample
                            )

                        bias += self.Learning_rate * self.C * y_binary[idx]

            self.classifiers[cls] = (weights, bias)

    def predict(self, X):
        scores = []

        for cls, (weights, bias) in self.classifiers.items():
            class_score = np.dot(X, weights) + bias
            scores.append(class_score)

        scores = np.array(scores)

        best_class_index = np.argmax(scores, axis=0)

        return np.array(list(self.classifiers.keys()))[best_class_index]


# 4) Baseline Multiclass Linear SVM
print("PHASE 2 BASELINE: Multiclass Linear SVM L2")

BASELINE_TRAIN_SIZE = 50000

X_tr_use = X_train[:BASELINE_TRAIN_SIZE]
y_tr_use = y_train[:BASELINE_TRAIN_SIZE]

svm_base = linearSVM(
    C=0.1,
    Learning_rate=0.001,
    n_epochs=20,
    regularization="l2"
)

svm_base.fit(X_tr_use, y_tr_use)

val_pred_base = svm_base.predict(X_val)
base_matrix = print_results(
    "Validation Results Baseline L2 Linear SVM",
    y_val,
    val_pred_base
)


# 5) Improvement 1: L1 vs L2 Regularization
print("Improvement 1: L1 vs L2 Regularization")

regularization_types = ["l2", "l1"]

for reg in regularization_types:
    print(f"\nTraining with {reg.upper()} regularization...")

    svm_reg = linearSVM(
        C=0.1,
        Learning_rate=0.001,
        n_epochs=20,
        regularization=reg
    )

    svm_reg.fit(X_tr_use, y_tr_use)

    val_pred = svm_reg.predict(X_val)

    print_results(
        f"Validation Results {reg.upper()} Regularization",
        y_val,
        val_pred
    )


# 6) Improvement 2: Learning Curves underfitting/overfitting 
print("Improvement 2: Learning Curves")

train_sizes = [500, 1000, 3000, 5000, 10000]

train_accs = []
val_accs = []

for size in train_sizes:
    print(f"\nTraining with {size} samples...")

    svm_lc = linearSVM(
        C=0.1,
        Learning_rate=0.001,
        n_epochs=10,
        regularization="l2"
    )

    svm_lc.fit(X_train[:size], y_train[:size])

    train_pred = svm_lc.predict(X_train[:size])
    val_pred = svm_lc.predict(X_val)

    train_acc, _, _, _, _ = evaluate_model(y_train[:size], train_pred)
    val_acc, _, _, _, _ = evaluate_model(y_val, val_pred)

    train_accs.append(train_acc * 100)
    val_accs.append(val_acc * 100)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Val Accuracy  : {val_acc * 100:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_accs, marker="o", label="Training Accuracy")
plt.plot(train_sizes, val_accs, marker="s", label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy (%)")
plt.title("Learning Curves - Multiclass Linear SVM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 7) Improvement 3: Bias-Variance Analysis using C values
print("Improvement 3: Bias-Variance Analysis Using C Values")

C_values = [0.001, 0.01, 0.1, 1.0]

c_train_accs = []
c_val_accs = []

for c in C_values:
    print(f"\nTraining with C = {c}...")

    svm_c = linearSVM(
        C=c,
        Learning_rate=0.001,
        n_epochs=10,
        regularization="l2"
    )

    svm_c.fit(X_train[:3000], y_train[:3000])

    train_pred = svm_c.predict(X_train[:3000])
    val_pred = svm_c.predict(X_val)

    train_acc, _, _, _, _ = evaluate_model(y_train[:3000], train_pred)
    val_acc, _, _, _, _ = evaluate_model(y_val, val_pred)

    c_train_accs.append(train_acc * 100)
    c_val_accs.append(val_acc * 100)

    print(f"C = {c}")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Val Accuracy  : {val_acc * 100:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(C_values, c_train_accs, marker="o", label="Training Accuracy")
plt.plot(C_values, c_val_accs, marker="s", label="Validation Accuracy")
plt.xscale("log")
plt.xlabel("C Value")
plt.ylabel("Accuracy (%)")
plt.title("Bias-Variance Analysis Effect of C")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 8) Improvement 4: Hyperparameter Tuning 3 fold cross-validation
print("Improvement 4: Hyperparameter tuning with 3 fold cross-validation")

TUNE_SIZE = 3000

X_tune = X_train[:TUNE_SIZE]
y_tune = y_train[:TUNE_SIZE]

def create_folds(num_samples, k=3):
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    folds = np.array_split(indices, k)

    return folds


experiments = [
    {"C": 0.01, "Learning_rate": 0.001,  "regularization": "l2"},
    {"C": 0.1,  "Learning_rate": 0.001,  "regularization": "l2"},
    {"C": 1.0,  "Learning_rate": 0.001,  "regularization": "l2"},
    {"C": 0.1,  "Learning_rate": 0.0001, "regularization": "l2"},
    {"C": 0.1,  "Learning_rate": 0.001,  "regularization": "l1"},
]

folds = create_folds(len(X_tune), k=3)

best_f1 = -1
best_params = None

for exp in experiments:
    print("\nTesting parameters:")
    print(exp)

    fold_f1_scores = []

    for fold_number in range(3):
        val_indices = folds[fold_number]

        train_indices = []

        for i in range(3):
            if i != fold_number:
                train_indices.extend(folds[i])

        train_indices = np.array(train_indices)

        X_fold_train = X_tune[train_indices]
        y_fold_train = y_tune[train_indices]

        X_fold_val = X_tune[val_indices]
        y_fold_val = y_tune[val_indices]

        svm_cv = linearSVM(
            C=exp["C"],
            Learning_rate=exp["Learning_rate"],
            n_epochs=5,
            regularization=exp["regularization"]
        )

        svm_cv.fit(X_fold_train, y_fold_train)

        fold_pred = svm_cv.predict(X_fold_val)

        _, _, _, fold_f1, _ = evaluate_model(y_fold_val, fold_pred)

        fold_f1_scores.append(fold_f1)

        print(f"  Fold {fold_number + 1} F1-score: {fold_f1 * 100:.2f}%")

    avg_f1 = np.mean(fold_f1_scores)

    print(f"Average F1-score: {avg_f1 * 100:.2f}%")

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_params = exp


print("\nBest Hyperparameters:")
print(best_params)
print(f"Best Cross-Validation F1-score: {best_f1 * 100:.2f}%")


# 9) Final Model using Best Hyperparameters
print("Final model evaluation on test set")

FINAL_TRAIN_SIZE = 50000

final_model = linearSVM(
    C=best_params["C"],
    Learning_rate=best_params["Learning_rate"],
    n_epochs=20,
    regularization=best_params["regularization"]
)

final_model.fit(X_train[:FINAL_TRAIN_SIZE], y_train[:FINAL_TRAIN_SIZE])

test_pred = final_model.predict(X_test)

test_matrix = print_results(
    "Final test results: Best Multiclass Linear SVM",
    y_test,
    test_pred
)

draw_confusion_matrix(
    test_matrix,
    "Final multiclass linear SVM confusion matrix"
)