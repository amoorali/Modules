import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class BaseModel(ABC):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        """Initializes with learning rate and number of epochs.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): Number of training iterations.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def initialize_weights(self, input_size: int, method: str = 'xavier') -> None:
        def zero():
            """Initialize weights to zero."""
            self.weights = np.zeros(input_size)  # Random normal distribution
            self.bias = np.zeros()  # Random bias initialization

        def normal():
            """Randomly initialize weights using a normal distribution."""
            self.weights = np.random.randn(input_size)  # Random normal distribution
            self.bias = np.random.randn()  # Random bias initialization
        
        def uniform():
            """Randomly initialize weights using a uniform distribution."""
            self.weights = np.random.uniform(-0.5, 0.5, input_size)  # Random uniform distribution
            self.bias = np.random.uniform(-0.5, 0.5)  # Random bias initialization

        def xavier():
            """Xavier (Glorot) initialization for weights."""
            limit = np.sqrt(6 / (input_size + 1))  # Xavier initialization formula
            self.weights = np.random.uniform(-limit, limit, input_size)
            self.bias = 0  # Bias is typically initialized to 0

        getattr(self, method, xavier)()

    def linear(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes the linear activation function.

        The linear activation function is defined as f(x) = x, meaning 
        the output is the same as the input without any transformation.

        Args:
            z (float): The input value.

        Returns:
            float: The same value as the input.
        """
        return z

    def relu(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Applies the ReLU (Rectified Linear Unit) activation function.

        The function returns the input value if it is positive; otherwise,
        it returns zero.

        Args:
            z (float or np.ndarray): The input value or array.

        Returns:
            float or np.ndarray: The transformed value after applying ReLU.
        """
        return np.maximum(0, z)
    
    def leaky_relu(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Applies the Leaky ReLU activation function.

        Similar to ReLU, but allows small negative values to avoid dead neurons.

        Args:
            z (float or np.ndarray): The input value or array.

        Returns:
            float or np.ndarray: The transformed value after applying Leaky ReLU.
        """
        return np.maximum(0.01 * z, z)

    def sigmoid(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes the sigmoid activation function.

        The sigmoid function maps any real number into the range (0,1),
        making it useful for probability estimation.

        Args:
            z (float or np.ndarray): The input value or array.

        Returns:
            float or np.ndarray: The sigmoid-activated value.
        """
        return 1 / (1 + np.exp(-z))

    def tanh(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes the hyperbolic tangent (tanh) activation function.

        The tanh function maps input values to the range (-1, 1),
        offering stronger gradients than sigmoid.

        Args:
            z (float or np.ndarray): The input value or array.

        Returns:
            float or np.ndarray: The tanh-activated value.
        """
        return np.tanh(z)
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of each row of the input array (logits) to obtain class probabilities.

        The softmax function converts raw scores (logits) into probabilities
        by exponentiating the logits and normalizing them across all classes,
        so the sum of probabilities for each sample is 1.

        Args:
            z (np.ndarray): A 2D array where each row represents the raw scores (logits) for a sample.
                            Shape (m, k), where m is the number of samples and k is the number of classes.

        Returns:
            np.ndarray: A 2D array of the same shape (m, k) where each row contains the class probabilities
                        obtained by applying softmax to the logits of each sample.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilize with np.max
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @abstractmethod
    def forward(self, X: np.ndarray, activation: str = 'sigmoid') -> np.ndarray:
        """Compute model output - to be implemented in subclasses."""
        pass

    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the loss function - to be implemented in subclasses."""
        pass

    @abstractmethod
    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        """Compute gradients and update weights - to be implemented in subclasses."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the model.

        Initializes weights and bias, then iterates over a fixed number of epochs, updating weights at each step.

        Args:
            X (np.ndarray): Input feature matrix of shape (m, n).
            y (np.ndarray): True labels of shape (m,).

        Prints:
            Cost value every 100 epochs to track training progress.

        Stores:
            self.loss_history (list): Stores cost per 100 epochs.
        """
        self.initialize_weights(X.shape[1])
        self.loss_history = []

        for i in range(self.epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make predictions using the trained model."""
        y_pred = self.forward(X)

        is_softmax = np.allclose(np.sum(y_pred, axis=0), 1)

        if is_softmax:
            # Raise an error if threshold is passed with softmax
            if threshold != 0.5:
                raise ValueError("Threshold is not applicable for softmax output!")
            return np.argmax(y_pred, axis=1)  # Class prediction for softmax

        else:  # Multi-class classification or binary classification
            return (y_pred >= threshold).astype(int)  # Thresholding for binary prediction