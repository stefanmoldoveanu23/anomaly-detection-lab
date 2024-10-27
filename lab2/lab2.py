import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

params = [5, 7, 3]
size = 100

def ex1(n_features=1):
    mean = np.random.uniform(low=-10, high=10, size=n_features + 1)
    variance = [0.0 for i in range(n_features + 1)]

    total_subplots = 2 ** (n_features + 1)
    it = bkt(0, mean, variance, n_features + 1)
    i = 0

    fig = plt.figure(figsize=(100, 100))

    for (x, y) in it:
        i += 1

        X = x.copy()
        X = np.append(X, np.array([y]).transpose(), axis = 1)

        U, _, _ = svd(X)
        U = np.diagonal(U)
        colors = ['b' if i < 0.99 else 'r' for i in U]

        if n_features == 1:
            x = x.transpose()[0]
            plt.subplot(1, total_subplots, i)
            plt.scatter(x, y, color = colors)
        else:
            ax = fig.add_subplot(1, total_subplots, i, projection='3d')
            ax.scatter3D(x[:,0], x[:,1], y, color=colors)

    plt.savefig("plot")

def bkt(i, mean, variance, n=2):
    print(i)
    if i == n:
        vals = []

        for j in range(n):
            vals.append(np.random.normal(loc=mean[j], scale=variance[j], size=size))

        x = np.atleast_2d(np.array(vals)).T
        y = np.sum(x[:,:-1] * params[:n - 1], axis=1) + params[n - 1] + x[:,-1:].transpose()[0]

        print(mean)
        print(variance)

        yield x[:,:-1], y
        return

    for j in range(2):
        if j == 0:
            variance[i] = np.random.uniform(low=0.0, high=0.1)
        else:
            variance[i] = np.random.uniform(low=0.5, high=1.0)

        it = bkt(i + 1, mean, variance, n)
        for x, y in it:
            yield x, y

def ex2(n_neighbors_iter=range(1, 2)):
    clusters = generate_data_clusters(n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1)

    for n_neighbors in n_neighbors_iter:
        model = KNN(contamination=0.1, n_neighbors=n_neighbors)

        model.fit(clusters[0])

        train_data = np.array(clusters[0])
        test_data = np.array(clusters[1])

        ground_truth_train = clusters[2]
        predicted_train = model.decision_function(clusters[0])
        ground_truth_test = clusters[3]
        predicted_test = model.decision_function(clusters[1])

        plt.subplot(2, 2, 1)
        plt.scatter(train_data[:,0], train_data[:,1], color=['red' if i == 1 else 'blue' for i in ground_truth_train])

        plt.subplot(2, 2, 2)
        plt.scatter(train_data[:,0], train_data[:,1], color=['red' if i > 0.5 else 'blue' for i in predicted_train])

        plt.subplot(2, 2, 3)
        plt.scatter(test_data[:,0], test_data[:,1], color=['red' if i == 1 else 'blue' for i in ground_truth_test])

        plt.subplot(2, 2, 4)
        plt.scatter(test_data[:,0], test_data[:,1], color=['red' if i > 0.5 else 'blue' for i in predicted_test])

        plt.savefig('knn_' + str(n_neighbors) + '.png')

def ex3(n_neighbors_iter=range(1, 2)):
    cluster_1 = make_blobs(n_samples=200, n_features=2, centers=[(-10,-10)], cluster_std=2)[0]
    cluster_2 = make_blobs(n_samples=100, n_features=2, centers=[(10,10)], cluster_std=6)[0]

    data = np.append(cluster_1, cluster_2, axis=0)

    for n_neighbors in n_neighbors_iter:
        model_knn = KNN(n_neighbors=n_neighbors, contamination=0.07)
        model_lof = LOF(n_neighbors=n_neighbors, contamination=0.07)

        model_knn.fit(data)
        model_lof.fit(data)

        predicted_knn = model_knn.decision_function(data)
        predicted_lof = model_lof.decision_function(data)

        plt.subplot(1, 2, 1)
        plt.scatter(data[:,0], data[:,1], color=['red' if i > 0.5 else 'blue' for i in predicted_knn])

        plt.subplot(1, 2 ,2)
        plt.scatter(data[:,0], data[:,1], color=['red' if i > 1.0 else 'blue' for i in predicted_lof])

        plt.savefig('versus_' + str(n_neighbors) + '.png')

def get_BA(predictions, ground_truth):
    TP, TN, FP, FN = 0, 0, 0, 0

    for (prediction, ground_truth) in list(zip(predictions, ground_truth)):
        if prediction == 1:
            if ground_truth == 1:
                TP += 1
            else:
                FP += 1
        else:
            if ground_truth == 1:
                FN += 1
            else:
                TN += 1
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    return (TPR + TNR) / 2

def ex4():
    dictionary = {}
    loadmat('cardio.mat', dictionary)

    data = dictionary['X']
    labels = dictionary['y']

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels)

    normalize(data_train)
    normalize(data_test)

    models = [LOF(n_neighbors=n_neighbors) for n_neighbors in range(30, 121, 9)]

    predictions_train = []
    predictions_test = []

    print( "Initial BAs:" )
    for model in models:
        model.fit(data_train)

        p_train = model.decision_function(data_train)
        p_test = model.decision_function(data_test)

        p_train = [1 if i > 1.0 else 0.0 for i in p_train]
        p_test = [1 if i > 1.0 else 0.0 for i in p_test]

        predictions_train.append(p_train)
        predictions_test.append(p_test)

        BA_train = get_BA(p_train, labels_train)
        BA_test = get_BA(p_test, labels_test)

        print(BA_train, BA_test)

    std_train = standardizer(predictions_train)
    std_test = standardizer(predictions_test)

    avg_train = average(std_train)
    max_train = maximization(std_train)

    print(avg_train, max_train)

    avg_test = average(std_test)
    max_test = maximization(std_test)

    print(avg_test, max_test)

# uncomment to run exercise 1
# pass parameter n_features: 1 for 2D plots, 2 for 3D plots
# default is 1
# ex1()

# uncomment to run exercise 2
# pass parameter n_neighbors_iter: default is range(1, 2)
# ex2(range(1, 6))

# uncomment to run exercise 3
# pass parameter n_neighbors_iter: default is range(1, 2)
# ex3(range(1, 6))

# uncomment to run exercise 4
# ex4()
