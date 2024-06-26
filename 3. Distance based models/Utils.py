import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from os.path import join
import csv
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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


def tune_gamma(X_train, X_test, y_train, y_test, gamma_values):
    best_accuracy = 0
    best_gamma = None

    for gamma in gamma_values:
        # Create a K-NN classifier with the custom Gaussian kernel
        knn_classifier = KNeighborsClassifier(
            n_neighbors=7, metric=custom_gaussian_kernel, metric_params={"gamma": gamma}
        )
        knn_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn_classifier.predict(X_test)

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Gamma = {gamma}, Accuracy = {accuracy:.2f}")

        # Update the best gamma if the current gamma gives higher accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_gamma = gamma

    print(f"Best Gamma = {best_gamma}, Best Accuracy = {best_accuracy:.2f}")

    return best_gamma


def custom_gaussian_kernel(x, y, gamma):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    euclidean_dist = euclidean_distances(x, y)
    return 1 - np.exp(-gamma * euclidean_dist**2)


def load_data(file_path, file_name):
    with open(join(file_path, file_name)) as csv_file:
        data_file = csv.reader(csv_file, delimiter=",")
        temp1 = next(data_file)
        n_samples = int(temp1[0])
        n_features = int(temp1[1])
        temp2 = next(data_file)
        feature_names = np.array(temp2[:n_features])

        data_list = [iter for iter in data_file]

        data = np.asarray(data_list, dtype=np.float64)

    return (data, feature_names, n_samples, n_features)


def get_best_shilouette(n, k, data):
    best_silhouette_score = -1

    for _ in range(n):
        # Esegui K-Means per il dataset 1
        kmeans = KMeans(n_clusters=k, random_state=None, n_init=10).fit(data)
        # Calcola l'indice di Silhouette per la soluzione corrente
        silhouette_score_current = silhouette_score(data, kmeans.labels_)

        # Verifica se questa soluzione ha un punteggio Silhouette migliore
        if silhouette_score_current > best_silhouette_score:
            best_silhouette_score = silhouette_score_current
            best_kmeans = kmeans
    return best_silhouette_score
