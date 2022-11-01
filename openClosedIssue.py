import pandas as pd
from setup.get_wine_data import DATA_PATH
from os import path as osp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class WinePreprocessor:
    def _separate_features(self, data):
        y = data['target']
        x = data.drop(columns='target')
        return x, y
    
    def min_max_scaler(self, data):
        x, _ = self._separate_features(data)
        print('Min Max Scaler preprocess')
        scaler = MinMaxScaler()
        return scaler.fit_transform(x)
    
    def standard_scaler(self, data):
        x, _ = self._separate_features(data)
        print('Standard Scaler preprocess')
        scaler = StandardScaler()
        return scaler.fit_transform(x)

    def normalizer(self, data):
        x, _ = self._separate_features(data)
        print('Normalize preprocess')
        normalizer = Normalizer()
        return normalizer.fit_transform(x)



class MLPipeline:
    def __init__(self, preprocessor, preporcess_type):
        self._preprocessor = preprocessor
        self._preprocess_type = preporcess_type

    def run(self, data):
        if self._preprocess_type == 'min_max':
            return self._preprocessor.min_max_scaler(data)
        elif self._preprocess_type == 'standard':
            return self._preprocessor.standard_scaler(data)
        elif self._preprocess_type == 'normalize':
            return self._preprocessor.normalizer(data)


if __name__ == '__main__':
    train_data = pd.read_csv(osp.join(DATA_PATH, 'wine_train.csv'))
    test_data = pd.read_csv(osp.join(DATA_PATH, 'wine_test.csv'))

    wine_preprocessor = WinePreprocessor()
    wine_pipeline = MLPipeline(wine_preprocessor, 'min_max')
    # wine_pipeline = MLPipeline(wine_preprocessor, 'standard')
    # wine_pipeline = MLPipeline(wine_preprocessor, 'normalize')
    
    X = wine_pipeline.run(train_data)
    print('Done!')