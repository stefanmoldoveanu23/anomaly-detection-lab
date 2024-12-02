from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, make_scorer
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def draw_subplots(X, y_true, y_pred, fig, column):
    ax = fig.add_subplot(2, 2, column * 2 + 1, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], color=['blue' if x == 0.0 else 'red' for x in y_true])

    ax = fig.add_subplot(2, 2, column * 2 + 2, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], color=['blue' if x == 0.0 else 'red' for x in y_pred])

def ex1():
    data = generate_data(n_train=300, n_test=200, n_features=3, contamination=0.15)

    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    tests = [
        (OCSVM(kernel='linear', contamination=0.3), 'ocsvm_linear.png', 1000.0),
        (OCSVM(kernel='rbf', contamination=0.3), 'ocsvm_rbf.png', 100.0),
        (DeepSVDD(n_features=3, contamination=0.3), 'deepsvdd.png', 1.0)
    ]

    for (model, filename, threshold) in tests:
        model.fit(X_train)
        y_pred = model.decision_function(X_test)

        y_pred_b = 1 * (y_pred > threshold)
        BA_score = balanced_accuracy_score(y_test, y_pred_b)
        RA_score = roc_auc_score(y_test, y_pred)

        print("Balanced accuracy:", BA_score)
        print("Roc auc accuracy:", RA_score)

        y_pred_train = 1 * (model.decision_function(X_train) > threshold)

        fig = plt.figure()
        draw_subplots(X_train, y_train, y_pred_train, fig, 0)
        draw_subplots(X_test, y_test, y_pred_b, fig, 1)

        plt.savefig(filename)

def my_score(y_true, y_pred):
    return balanced_accuracy_score(y_true * (-2) + 1, y_pred)

def ex2():
    data = {}
    loadmat('cardio.mat', data)
    
    X = data['X']
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4)

    my_scorer = make_scorer(my_score)
    pipe = Pipeline(
            [
                ("scaling", StandardScaler()),
                ("classify", OneClassSVM())
            ]
        )
    param_grid = [
        {
            'classify__kernel': ['linear'],
            'classify__nu': (0.5, 0.3, 0.8, 0.2)
        },
        {
            'classify__kernel': ['rbf', 'sigmoid'],
            'classify__gamma': ('scale', 0.2, 0.5),
            'classify__nu': (0.5, 0.3, 0.8, 0.2)
        }
    ]
    GS_model = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=my_scorer
            )

    GS_model.fit(X_train, y_train)
    print(GS_model.best_params_)
    print(GS_model.score(X_test, y_test))

def ex3():
    data = {}
    loadmat('shuttle.mat', data)

    X = data['X']
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    models = [
            (OCSVM(), 'OCSVM:', 1.0),
            (DeepSVDD(n_features=9), 'DeepSVDD basic:', 1.0),
            (DeepSVDD(n_features=9, hidden_neurons=[128, 64, 64, 32]), 'DeepSVDD custom1:', 0.02),
            (DeepSVDD(n_features=9, hidden_neurons=[256, 128, 64, 32], dropout_rate=0.25, epochs=200), 'Deep SVDD custom2:', 0.015)
    ]

    for (model, name, threshold) in models:
        model.fit(X_train)
        y_pred = model.decision_function(X_test)
        print(y_pred)
        y_pred_b = 1 * (y_pred > threshold)

        BA = balanced_accuracy_score(y_test, y_pred_b)
        RA = roc_auc_score(y_test, y_pred)

        print(name)
        print("BA:", BA)
        print("RA:", RA)

# remove comment to run exercise 1
# ex1()

# remove comment to run exercise 2
# ex2()

# remove comment to run exercise 3
# ex3()
