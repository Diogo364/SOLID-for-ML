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


class StandardScalerPreprocessor(PreprocessorInterface):
    def preprocess_data(self, data, with_mean):
        x, _ = self._separate_features(data)
        print('Standard Scaler preprocess')
        scaler = StandardScaler(with_mean=with_mean)
        return scaler.fit_transform(x)


class MinMaxPreprocessor(PreprocessorInterface):
    def preprocess_data(self, data, bottom_lim, top_lim):
        x, _ = self._separate_features(data)
        scaler = MinMaxScaler(feature_range=(bottom_lim, top_lim))
        return scaler.fit_transform(x)


class MLPipeline:
    def __init__(self, preprocessor):
        self._preprocessor = preprocessor

    def run(self, data, **kwargs):
        return self._preprocessor.preprocess_data(data, **kwargs)


if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    wine_preprocessor = MinMaxPreprocessor()
    # wine_preprocessor = StandardScalerPreprocessor()
    
    wine_pipeline = MLPipeline(wine_preprocessor)
    
    X = wine_pipeline.run(train_data, bottom_lim=-1, top_lim=1)
    print('Done!')