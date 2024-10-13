import pyod.utils.data as data;
import matplotlib.pyplot as plt;
from pyod.models.knn import KNN;
import sklearn.metrics as metrics;
from scipy.stats import uniform;
import numpy as np;
import math;

def ex12(contamination=0.1):
    gen_data = data.generate_data(400, 100, 2, contamination=contamination);
    #print( "Number of outliers:", sum(gen_data[2]) );

    X_train = [list(zip(*gen_data[0]))[0], list(zip(*gen_data[0]))[1]];
    c_train = list(map(lambda x: 'blue' if x == 0 else 'red', gen_data[2]));
    y_train = gen_data[2];

    X_test = [list(zip(*gen_data[1]))[0], list(zip(*gen_data[1]))[1]];
    y_test = gen_data[3];

    plt.scatter(X_train[0], X_train[1], c=c_train);
    plt.savefig("scatter.png");

    model = KNN(contamination=contamination);
    model.fit(gen_data[0]);

    y_train_pred = model.labels_;
    y_test_pred = model.predict(gen_data[1]);
    y_test_scores = model.decision_function(gen_data[1]);

    conf = metrics.confusion_matrix(y_test, y_test_pred);

    TP = conf[0][0];
    FP = conf[0][1];
    FN = conf[1][0];
    TN = conf[1][1];

    TPR = TP / (TP + FN);
    TNR = TN / (TN + FP);
    BA = (TPR + TNR) / 2;

    print( "Balanced accuracy:", BA );

    fpr, tpr, _ = metrics.roc_curve(y_test, y_test_scores);

    plt.clf();
    plt.plot(fpr, tpr);

    plt.savefig("roc_curve.png");

def gen_params(n=10):
    means = np.random.uniform(low=-50.0, high=50.0, size=n);
    covar = np.random.uniform(low=0.0, high=5.0, size=(n, n));
    covar = covar @ covar.transpose();

    return means, covar

def solve_upper_triangle(A, b):
    n = len(b);
    for i in range(n - 1, -1, -1):
        b[i] /= A[i, i];
        b[:i] -= A[:i, i] * b[i];

    return b

def solve_lower_triangle(A, b):
    n = len(b);
    for i in range(n):
        b[i] /= A[i, i];

        if i != n - 1:
            b[i + 1:] -= A[i + 1, i] * b[i];

    return b

def preprocess(A):
    n = len(A);
    B = np.copy(A);

    for i in range(n):
        B[i, i] = math.sqrt(B[i, i]);
        B[i + 1:, i] /= B[i, i];

        for j in range(i + 1, n):
            B[j:, j] -= B[j:, i] * B[j, i];

    return B

def solve_system(A, b):
    y = np.copy(b);

    y = solve_upper_triangle(np.transpose(A), y);
    y = solve_lower_triangle(A, y);

    return y;

def ex3(gen_data,  n=1000, d=1):
    # for i in range(10):
    #     counts, bins = np.histogram(gen_data[:, i]);
    #     plt.stairs(counts, bins);
    # plt.savefig("hist.png")
    return 1

def ex4(n=1000, d=10):
    (means, covar) = gen_params();
    gen_data = np.random.multivariate_normal(means, covar, size=n);

    quantile = np.quantile(gen_data, 0.9);
    print("Quantile: ", quantile);
    preprocessed = preprocess(covar);
    cnt = 0;
    
    for X in gen_data:
        y = X - means;
        Z_score = math.sqrt(np.sum(np.dot(y, solve_system(preprocessed, y))));
        #print(Z_score);
        if Z_score > quantile:
            cnt += 1;

    print(cnt);

# Remove comment to run exercises 1 and 2
# ex12();

# Remove comment to run exercise 4
ex4()
