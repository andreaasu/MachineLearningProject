import os
from src.data_loader import load_mnist
from src.logistic_regression import LogisticRegression
from src.utils import print_metrics, plot_confusion_matrix

def train_and_save():
    X_train, X_test, y_train, y_test = load_mnist()
    model = LogisticRegression(alpha=1.3, num_iter=15000)
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/final_logistic_regression_13_15000_0.4.npy'
    model.save_model(model_path)
    
    print("training complete and model saved.")
    print(f"iterations: {model.actual_iter}")
    if model.losses:
        print(f"Final Loss: {model.losses[-1]:.4f}")
    
    y_pred_train = model.predict(X_train, threshold=0.4)
    y_pred_test = model.predict(X_test,threshold=0.4)
    print_metrics(y_train, y_pred_train, y_test, y_pred_test, model)

if __name__ == "__main__":
    train_and_save()
