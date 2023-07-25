import numpy as np

class PlayTennis:
    def __init__(self, X, y):
        self.X = X
        self.y_unique = np.unique(y).tolist()
        self.y = y.tolist()
        self.n_sample = len(self.y)

    def compute_prior_probablity(self):
        prior_probability = np.zeros(len(self.y_unique))

        for i in range(len(self.y_unique)):
            prior_probability[i] = self.y.count(self.y_unique[i])/self.n_sample

        return prior_probability

    def compute_conditional_probability(self):
        conditional_probability = []
        list_x_name = []
        prior_probability = self.compute_prior_probablity()

        for i in range(self.X.shape[1]):
            x_unique = np.unique(self.X[:,i]).tolist()
            list_x_name.append(x_unique)
            x_conditional_probability = {label:[0]*len(x_unique) for label in self.y_unique}

            for j in range(len(self.y)):
                label = self.y[j]
                num_label = prior_probability[self.y_unique.index(label)]*self.n_sample
                x_conditional_probability[label][x_unique.index(self.X[j][i])] += 1/num_label

            conditional_probability.append(x_conditional_probability)

        return conditional_probability, list_x_name

    def prediction_play_tennis(self, X_new):
        conditional_probability, list_x_name = self.compute_conditional_probability()
        prior_probability = self.compute_prior_probablity()
        p_temp = [p for p in prior_probability]

        for i in range(len(X_new)):
            for j in range(len(p_temp)):
                p_temp[j] = p_temp[j]*conditional_probability[i][self.y_unique[j]][
                        np.where(X_new[i] == np.array(list_x_name[i]))[0][0]]

        mx = max(p_temp)
        sm = sum(p_temp)
        output = self.y_unique[p_temp.index(mx)]

        return f"Go or not? => {output}: probability={round(mx/sm, 2)}"