import numpy as np
import math

class IrisPredict:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_unique = np.unique(y_train).tolist()
        self.y_train = y_train.tolist()
        self.n_sample = len(self.y_train)

    def compute_prior_probablity(self):
        prior_probability = np.zeros(len(self.y_unique))

        for i in range(len(self.y_unique)):
            prior_probability[i] = self.y_train.count(self.y_unique[i])/self.n_sample

        return prior_probability

    def mean_variance(self):
        data_list = {label:[] for label in self.y_unique}
        mean_variance_dic = {label:[] for label in self.y_unique}

        for i in range(self.n_sample):
            data_list[self.y_train[i]].append(self.X_train[i])

        for j in data_list:
            data_array = np.array(data_list[j])

            for t in range(self.X_train.shape[1]):
                mean = sum(data_array[:,t])/len(data_array)
                deviations = [(k - mean) ** 2 for k in data_array[:,t]]
                variance = sum(deviations)/len(data_array)
                mean_variance_dic[j].append((mean, variance))

        return mean_variance_dic

    def prediction_iris(self, X_new):
        mean_variance_dic = self.mean_variance()
        prior_probability = self.compute_prior_probablity()
        p_temp = [p for p in prior_probability]

        for i in range(len(X_new)):
            for j in range(len(p_temp)):
                mean_ij, variance_ij = mean_variance_dic[self.y_unique[j]][i]
                p_temp[j] = p_temp[j]*(1/(2*math.pi*variance_ij)**0.5)*math.exp(-(X_new[i] - mean_ij)**2/(2*variance_ij))

        mx = max(p_temp)
        sm = sum(p_temp)
        output = self.y_unique[p_temp.index(mx)]

        return f"Which flower? => {output}: probability={mx/sm}"