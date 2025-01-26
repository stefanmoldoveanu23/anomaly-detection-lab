import numpy as np
import time
from Utils.Data import Data
from Utils.Forest import Forest
from sklearn.metrics import roc_auc_score
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with np.load('../../OptIForest/data/35_SpamBase.npz') as dictionary:
        X = dictionary['X']
        y_true = dictionary['y']

        start_time = time.time()
        iForest = Forest(data=Data(X, np.array([i for i in range(X.shape[0])])), bucket_size=4, n_trees=1)
        mid_time = time.time()
        print(f"Build time: {mid_time - start_time}")
        y_pred = iForest.predict(X)
        end_time = time.time()
        print(f"Prediction time: {end_time - mid_time}")
        print(f"Score: {roc_auc_score(y_true, y_pred)}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
