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
    def __init__(self, with_mean):
        self.with_mean = with_mean

    def preprocess_data(self, data):
        x, _ = self._separate_features(data)
        print('Standard Scaler preprocess')
        scaler = StandardScaler(with_mean=self.with_mean)
        return scaler.fit_transform(x)


class MinMaxPreprocessor(PreprocessorInterface):
    def __init__(self, bottom_lim, top_lim):
        self.bottom_lim = bottom_lim
        self.top_lim = top_lim        
    
    def preprocess_data(self, data):
        x, _ = self._separate_features(data)
        scaler = MinMaxScaler(feature_range=(self.bottom_lim, self.top_lim))
        return scaler.fit_transform(x)


class MLPipeline:
    def __init__(self, preprocessor):
        self._preprocessor = preprocessor

    def run(self, data):
        return self._preprocessor.preprocess_data(data)


if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    wine_preprocessor = MinMaxPreprocessor(bottom_lim=-1, top_lim=1)
    # wine_preprocessor = StandardScalerPreprocessor(with_mean=True)
    
    wine_pipeline = MLPipeline(wine_preprocessor)
    
    X = wine_pipeline.run(train_data)
    print('Done!')