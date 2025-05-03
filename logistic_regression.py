import numpy as np
from Modules.base_model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        super().__init__(learning_rate, epochs)

    def forward(self, X: np.ndarray, activation: str = 'sigmoid') -> np.ndarray:
        """Compute predictions using the specified activation function.

        Args:
            X (np.ndarray): Input feature matrix of shape (m, n).
            activation (str, optional): Activation function to use (default is 'sigmoid').

        Returns:
            np.ndarray: Predicted values of shape (m, n).
        """
        # Check if the provided activation function is valid
        activation_func = getattr(self, activation, None)
        if activation_func is None:
            raise ValueError(f"Activation function '{activation}' is not defined. Using 'sigmoid' as default.")
        y_pred = activation_func(np.dot(X, self.weights) + self.bias)

        return y_pred

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the binary cross-entropy loss.

        Args:
            y_true (np.ndarray): True labels (0 or 1).
            y_pred (np.ndarray): Predicted probabilities (between 0 and 1).

        Returns:
            float: The computed cost (loss).
        """
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping to prevent log(0)
        
        m = len(y_true)
        return - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Performs one step of gradient descent to update model weights.

        Args:
            X (np.ndarray): Input feature matrix of shape (m, n).
            y_true (np.ndarray): True labels of shape (m,).
            y_pred (np.ndarray): Predicted labels of shape (m,).

        Updates:
            self.weights (np.ndarray): Updated weight vector.
            self.bias (float): Updated bias term.
        """
        m, n = X.shape

        dw = (1 / m) * np.dot(X.T, (y_pred - y_true))
        db = (1 / m) * np.sum(y_pred - y_true)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db


# Let's initialize with normal distribution
X_train = np.array([[0], [1], [2], [3], [4], [5]])
y_train = np.array([0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)