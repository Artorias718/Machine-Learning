import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def find_best_k(X_train, y_train):
    # Define the range of k values to search
    param_grid = {"n_neighbors": np.arange(1, 50)}

    # Create a k-NN classifier
    knn = KNeighborsClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Return the best parameter and its corresponding score
    best_k = grid_search.best_params_["n_neighbors"]
    best_score = grid_search.best_score_
    return best_k
