import os
import os.path as osp
from config import DATA_PATH
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    os.makedirs(DATA_PATH, exist_ok=True)
    data = load_wine(as_frame=True)

    train, test = train_test_split(data['frame'], test_size=0.25)

    train.to_csv(osp.join(DATA_PATH, 'wine_train.csv'), index=False)
    test.to_csv(osp.join(DATA_PATH, 'wine_test.csv'), index=False)
