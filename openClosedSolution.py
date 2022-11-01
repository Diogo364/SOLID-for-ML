import pandas as pd
from setup.get_wine_data import DATA_PATH
from os import path as osp
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class PreprocessorInterface(ABC):
    def _separate_features(self, data):
        y = data['target']
        x = data.drop(columns='target')
        return x, y

    @abstractmethod
    def preprocess_data(self, data):
        pass



class MinMaxPreprocessor(PreprocessorInterface):
    def preprocess_data(self, data):
        x, _ = self._separate_features(data)
        print('Min Max Scaler preprocess')
        scaler = MinMaxScaler()
        return scaler.fit_transform(x)


class StandardScalerPreprocessor(PreprocessorInterface):
    def preprocess_data(self, data):
        x, _ = self._separate_features(data)
        print('Standard Scaler preprocess')
        scaler = StandardScaler()
        return scaler.fit_transform(x)


class NormalizerPreprocessor(PreprocessorInterface):
    def preprocess_data(self, data):
        x, _ = self._separate_features(data)
        print('Standard Scaler preprocess')
        normalizer = Normalizer()
        return normalizer.fit_transform(x)



class MLPipeline:
    def __init__(self, preprocessor):
        self._preprocessor = preprocessor

    def run(self, data):
        return self._preprocessor.preprocess_data(data)


if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    wine_preprocessor = MinMaxPreprocessor()
    # wine_preprocessor = StandardScalerPreprocessor()
    # wine_preprocessor = NormalizerPreprocessor()
    
    wine_pipeline = MLPipeline(wine_preprocessor)
    
    X = wine_pipeline.run(train_data)
    print('Done!')