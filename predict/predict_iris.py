import pandas as pd
import numpy as np
from utils.Iris import IrisPredict

if __name__ == "__main__":
    data = pd.read_csv("..\datasets\iris.data.txt")
    data = np.array(data)
    X_train = data[:,:-1]
    y_train = data[:,-1]

    iris = IrisPredict(X_train, y_train)
    output = iris.prediction_iris([5.0, 2.2, 3.3, 0.9])
    print(output)