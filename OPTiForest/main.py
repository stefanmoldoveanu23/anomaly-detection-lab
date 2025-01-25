import numpy as np
import time
from Utils.Data import Data
from Utils.Forest import Forest
from sklearn.metrics import roc_auc_score

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with np.load('./Data/32_shuttle.npz') as dictionary:
        X = dictionary['X']
        y_true = dictionary['y']

        start_time = time.time()
        iForest = Forest(data=Data(X, np.array([i for i in range(X.shape[0])])), bucket_size=4, n_trees=100)
        mid_time = time.time()
        y_pred = iForest.predict(X)
        end_time = time.time()
        print(f"Build time: {mid_time - start_time}")
        print(f"Prediction time: {end_time - mid_time}")
        print(f"Score: {roc_auc_score(y_true, y_pred)}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
