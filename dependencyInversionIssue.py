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
        self.model = DecisionTreeClassifier(random_state=42)
    
    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class ClassifierEvaluator:
    def get_accuracy(self, y, yhat):
        return accuracy_score(y, yhat)


class MLPipeline:
    def __init__(self):
        self.preprocessor = WinePreprocessor()
        self.model = WineClassifier()
        self.evaluator = ClassifierEvaluator()

    def run(self, train_data, test_data):
        x_train, y_train = self.preprocessor.preprocess(train_data)
        self.model.train(x_train, y_train)
        
        x_test, y_test = self.preprocessor.preprocess(test_data, train=False)
        yhat = self.model.predict(x_test)
        
        return self.evaluator.get_accuracy(y_test, yhat)


if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    pipeline = MLPipeline()
    accuracy = pipeline.run(train_data, test_data)
    
    print(f"Model's accuracy: {accuracy}")