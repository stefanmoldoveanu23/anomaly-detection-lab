import numpy as np
from scipy.io import loadmat
from Utils.Data import Data
from Utils.Forest import Forest
from sklearn.metrics import roc_auc_score

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with np.load('./Data/34_smtp.npz') as dictionary:
        X = dictionary['X']
        y_true = dictionary['y']

        iForest = Forest(data=Data(X, np.array([i for i in range(X.shape[0])])), bucket_size=4, n_trees=100)
        y_pred = iForest.predict(X)
        print(y_true, y_pred, roc_auc_score(y_true, y_pred))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
