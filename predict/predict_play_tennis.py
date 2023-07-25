from utils.PlayTennis import PlayTennis
import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv("..\datasets\play_tennis_data.csv")
    data = np.array(data)
    X_data = data[:,:-1]
    y_data = data[:,-1]

    pt = PlayTennis(X_data, y_data)
    output = pt.prediction_play_tennis(['Sunny', 'Cool', 'High', 'Strong'])
    print(output)