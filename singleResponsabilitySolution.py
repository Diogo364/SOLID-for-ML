import pandas as pd
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


class WineClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier()
    
    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class ClassifierEvaluator:
    def evaluate(self, y, yhat):
        return accuracy_score(y, yhat)


if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    wine_preprocessor = WinePreprocessor()
    wine_classifier = WineClassifier()
    classifier_evaluator = ClassifierEvaluator()
    
    x_train, y_train = wine_preprocessor.preprocess(train_data)
    wine_classifier.train(x_train, y_train)
    
    x_test, y_test = wine_preprocessor.preprocess(test_data, train=False)
    yhat = wine_classifier.predict(x_test)
    
    accuracy = classifier_evaluator.evaluate(y_test, yhat)
    
    print(f"Model's accuracy: {accuracy}")