from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from random import choice
from setup.get_wine_data import DATA_PATH
from os import path as osp
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


class WinePreprocessor:
    def __init__(self):
        self._scaler = MinMaxScaler()

    def preprocess(self, data, train=True):
        y = data['target']
        x = data.drop(columns='target')
        if train:
            self._scaler.fit(x)
        
        return self._scaler.transform(x), y


class MLModelInterface(ABC):
    @abstractmethod
    def train(self, x, y):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def random_predict(self, x):
        pass


class WineEnsembleClassifier(MLModelInterface):
    def __init__(self, n_models=10):
        self.model = [DecisionTreeClassifier() for _ in range(n_models)]
        self.n_models = n_models

    def train(self, x, y):
        for model in self.model:
            model.fit(x, y)
    
    def predict(self, x):
        total = np.zeros(x.shape[0])
        for model in self.model:
            total += model.predict(x)
        return total/self.n_models
    
    def random_predict(self, x):
        random_model = choice(self.model)
        return random_model.predict(x)

class WineClassifier(MLModelInterface):
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
    
    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def random_predict(self, x):
        raise Exception('There is only one model inside this class')

if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    wine_preprocessor = WinePreprocessor()
    wine_classifier = WineEnsembleClassifier()
    
    x_train, y_train = wine_preprocessor.preprocess(train_data)
    wine_classifier.train(x_train, y_train)
    
    x_test, y_test = wine_preprocessor.preprocess(test_data, train=False)
    yhat = wine_classifier.predict(x_test)
    print('Done!')