import pandas as pd
from setup.get_wine_data import DATA_PATH
from os import path as osp
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

class WineClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self._scaler = MinMaxScaler()
    
    def preprocess(self, data, train=True):
        y = data['target']
        x = data.drop(columns='target')
        if train:
            self._scaler.fit(x)
        
        return self._scaler.transform(x), y

    def train(self, data):
        x, y = self.preprocess(data)
        self.model.fit(x, y)

    def evaluate(self, data):
        x, y = self.preprocess(data, train=False)
        yhat = self.model.predict(x)
        return accuracy_score(y, yhat)


if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    wine_classifier = WineClassifier()
    wine_classifier.train(train_data)
    accuracy = wine_classifier.evaluate(test_data)
    
    print(f"Model's accuracy: {accuracy}")