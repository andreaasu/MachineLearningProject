import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def accuracy(y_true, y_pred):
    # fraction of predictions that match the true labels
    return np.mean(y_true == y_pred)

def precision_recall_f1_per_class(cm):
    # cm rows = actual class, cols = predicted class
    num_classes = cm.shape[0]
    precisions = []
    recalls = []
    f1s = []
    for class_idx in range(num_classes):
        # tp = correct predictions for this class
        tp = cm[class_idx, class_idx]
        # fp = predicted this class but actually something else
        fp = cm[:, class_idx].sum() - tp
        # fn = actually this class but predicted something else
        fn = cm[class_idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s

def print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model):
    # figure out how many classes from the labels
    all_labels = np.concatenate([y_train, y_val, y_test])
    num_classes = len(np.unique(all_labels))

    for name, y_true, y_pred in [("Train", y_train, y_pred_train),
                                   ("Val", y_val, y_pred_val),
                                   ("Test", y_test, y_pred_test)]:
        acc = accuracy(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        precs, recs, f1s = precision_recall_f1_per_class(cm)

        print(f"{name} Accuracy: {acc:.4f}")

        if num_classes == 2:
            # for binary, show metrics for the positive class (index 1)
            print(f"{name} Precision: {precs[1]:.4f}")
            print(f"{name} Recall: {recs[1]:.4f}")
            print(f"{name} F1 Score: {f1s[1]:.4f}")
        else:
            # for multiclass, average across all classes
            print(f"{name} Precision (macro): {np.mean(precs):.4f}")
            print(f"{name} Recall (macro): {np.mean(recs):.4f}")
            print(f"{name} F1 Score (macro): {np.mean(f1s):.4f}")

    plot_confusion_matrix(y_test, y_pred_test, num_classes)
    plot_loss_curves(model)

def plot_confusion_matrix(y_true, y_pred, num_classes=2):
    # build confusion matrix using sklearn
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    # write the count in each cell
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.show()

def plot_loss_curves(model):
    # plot training loss over iterations
    plt.plot(model.losses, label=f"train alpha={model.alpha} iter={model.num_iter}")
    if model.val_losses and len(model.val_losses) > 0:
        # plot validation loss too if available
        plt.plot(model.val_losses, label=f"val alpha={model.alpha}")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("loss curve")
    plt.tight_layout()
    plt.show()
