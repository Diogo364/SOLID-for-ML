import os
from os import path as osp

root_path = '.'
while 'setup' not in os.listdir(root_path):
    root_path = osp.join(osp.pardir, root_path)

DATA_PATH = osp.join(root_path, 'data')
