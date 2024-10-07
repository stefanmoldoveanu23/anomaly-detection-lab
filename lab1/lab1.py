import pyod.utils.data as data;
import matplotlib.pyplot as plt;
from pyod.models.knn import KNN;
import sklearn.metrics as metrics;
from scipy.stats import uniform;
from math import *;

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

def normal_gen(distributions, n=1000):
    vals = [0 for i in range(n)];
    for i in range(n):
        val = [0 for j in range(len(distributions))];

        for j in range(len(distributions)):
            mu = distributions[j][0];
            var = distributions[j][1];
            U = uniform.rvs(size=1)[0];
            V = uniform.rvs(size=1)[0];

            val[j] = sqrt(-2 * log(U)) * sin(2 * pi * V);
            val[j] = mu + val[j] * var;
            vals[i] = val;

    return vals

def ex34(data):
    vals = generator([(20, 5), (10, 6), (0, 2)]);
    print(vals);



# Remove comment to run exercises 1 and 2
# ex12();

ex34(normal_gen);
