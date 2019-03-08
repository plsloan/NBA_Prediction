import numpy
import pandas
from sklearn.neighbors import KNeighborsClassifier

class KNN_Learner:
    k = 3
    model = KNeighborsClassifier(n_neighbors=k)

    def __init__(self, k_num=3):
        self.k = k_num
    def train(self, data_set, result_set):
        self.model.fit(data_set, result_set)
    def predict(self, data_set):
        return self.model.predict(data_set)
    def getModel(self):
        return self.model
