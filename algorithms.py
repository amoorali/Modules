import numpy as np

def pca_transition(X, n_components):
    """Implements PCA from scratch.

    Arguments:
    - X: NumPy array of shape (n_samples, n_features), the dataset.
    - n_components: Number of principal components to keep.

    Returns:
    - X_pca: Transformed data of shape (n_samples, n_components).
    - explained_variance_ratio: How much variance each component explains.
    - eigenvectors: The principal components.
    """
    # Step 1: Mean Centering
    X_meaned = X - np.mean(X, axis=0)

    # Step 2: Compute Covariance Matrix
    covariance_matrix = np.cov(X_meaned, rowvar=False)  # (n_features, n_features)

    # Step 3: Compute Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort Eigenvalues & Select Top Components
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]  # Sort eigenvectors

    # Step 5: Keep Only the Top 'n_components'
    top_eigenvectors = eigenvectors[:, :n_components]

    # Step 6: Project Data Onto New Basis
    X_pca = np.dot(X_meaned, top_eigenvectors)

    # Compute Explained Variance Ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_pca, explained_variance_ratio, top_eigenvectors