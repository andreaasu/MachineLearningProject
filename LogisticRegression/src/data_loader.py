from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
# X, y = mnist["data"], mnist["target"]

# def preprocess(X, y):
#     return (X / 255.0).astype(float), y.astype(int)

def load_mnist(): #to avoid local variable error
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist["data"], mnist["target"]
    X, y = (X / 255.0).astype(float), (y=='8').astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)