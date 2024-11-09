from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def predict(data, vectors, histograms, rge):
    projected = []

    for i in range(len(vectors)):
        projected.append(np.dot(data, vectors[i]))
    projected = np.array(projected).T

    scores = np.zeros((len(data), len(vectors)))
    for i in range(len(data)):
        projected[i] = np.array((projected[i] - rge[0]) / (rge[1] - rge[0]) * len(histograms[0]), dtype=np.int32)

        for j in range(len(vectors)):
            k = min(int(projected[i][j]), len(histograms[0]) - 1)
            scores[i][j] = histograms[j][k]

    return np.mean(scores, axis=1)


def ex1():
    data = np.array(make_blobs(n_samples=500, n_features=2)[0])

    vectors = np.random.multivariate_normal(np.zeros(10), np.identity(10))
    vectors = np.array([vectors[i::5] for i in range(5)])
    vectors = normalize(vectors)

    projected = []

    for i in range(5):
        projected.append(np.dot(data, vectors[i]))
    
    projected_arr = np.array(projected)

    range_min = np.min(projected)
    range_max = np.max(projected)

    for bins in range(10, 20, 2):
        projected = np.copy(projected_arr)

        histograms = []
        for i in range(5):
            histograms.append(np.histogram(projected[i], range=(range_min, range_max), bins=bins)[0] / 500.0)

        if bins == 12:
            print(histograms)

        projected = projected.T
        histograms = np.array(histograms)
        scores = np.zeros((500, 5))

        for i in range(500):
            projected[i] = np.array((projected[i] - range_min) / (range_max - range_min) * bins, dtype=np.int32)

            for j in range(5):
                k = min(int(projected[i][j]), bins - 1)
                scores[i][j] = histograms[j][k]

        scores = np.mean(scores, axis=1)
        plt.scatter(data[:,0], data[:,1], color=['blue' if x > 0.2 else 'red' for x in scores])

        plt.savefig('fig_fit_' + str(bins) + '.png')

        data_test = np.random.uniform(low=-3.0, high=3.0, size=(500, 2))
        scores_test = predict(data_test, vectors, histograms, (range_min, range_max))

        plt.clf()
        plt.scatter(data_test[:,0], data_test[:,1], color=['blue' if x > 0.2 else 'red' for x in scores_test])
        plt.savefig('fig_test_' + str(bins) + '.png')
        plt.clf()

def ex2():
    blobs = make_blobs(n_samples=[500, 500], n_features=2, centers=[(10, 0), (0, 10)], cluster_std=1.0)

    data = blobs[0]

    model_iforest = IForest(contamination=0.02)
    model_iforest.fit(data)

    model_dif = DIF(contamination=0.02, hidden_neurons=[128, 64, 32])
    model_dif.fit(data)

    model_loda = LODA(contamination=0.02, n_bins=15)
    model_loda.fit(data)

    test_data = np.random.uniform(low=-10, high=20, size=(1000, 2))

    prediction_iforest = model_iforest.decision_function(test_data)
    prediction_dif = model_dif.decision_function(test_data)
    prediction_loda = model_loda.decision_function(test_data)

    plt.subplot(1, 3, 1)
    plt.scatter(test_data[:,0], test_data[:,1], c=prediction_iforest)

    plt.subplot(1, 3, 2)
    plt.scatter(test_data[:,0], test_data[:,1], c=prediction_dif)

    plt.subplot(1, 3, 3)
    plt.scatter(test_data[:,0], test_data[:,1], c=prediction_loda)

    plt.savefig('iforest.png')

def ex2_3d():
    blobs = make_blobs(n_samples=[500, 500], n_features=3, centers=[(0, 10, 0), (10, 0, 10)])

    data = blobs[0]

    model_iforest = IForest(contamination=0.02)
    model_iforest.fit(data)

    model_dif = DIF(contamination=0.02, hidden_neurons=[128, 64, 32])
    model_dif.fit(data)

    model_loda = LODA(contamination=0.02, n_bins=15)
    model_loda.fit(data)

    test_data = np.random.uniform(low=-10, high=20, size=(1000, 3))

    prediction_iforest = model_iforest.decision_function(test_data)
    prediction_dif = model_dif.decision_function(test_data)
    prediction_loda = model_loda.decision_function(test_data)

    fig = plt.figure(figsize=(40, 40))

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c=prediction_iforest)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c=prediction_dif)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c=prediction_loda)

    plt.savefig('iforest3D.png')

def get_BA_RA(model, X_train, X_test, y_test, barrier):
    model.fit(X_train)
    
    predictions = model.decision_function(X_test)
    predictions_binary = np.array(list(map(lambda x: 1 if x > barrier else 0, predictions)))

    return balanced_accuracy_score(y_test, predictions_binary), roc_auc_score(y_test, predictions)

def ex3():
    data = loadmat('shuttle.mat')

    X = data['X']
    y = data['y']

    X = normalize(X)

    split = 0.1
    while split < 0.91:
        print("Test size: " + str(split * 100) + "%")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

        BA_iforest, RA_iforest = get_BA_RA(IForest(), X_train, X_test, y_test, 0.0)
        BA_dif, RA_dif = get_BA_RA(DIF(), X_train, X_test, y_test, 0.3)
        BA_loda, RA_loda = get_BA_RA(LODA(), X_train, X_test, y_test, 0.03)

        print(BA_iforest, RA_iforest)
        print(BA_dif, RA_dif)
        print(BA_loda, RA_loda)
        split = split + 0.1

# ex1()

# ex2()
# ex2_3d()

ex3()
