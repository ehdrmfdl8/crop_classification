import os
import math
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

CSV_EXTENSIONS = ['.csv', '.CSV']

def get_csv_paths(dataroot):
    paths = None # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_csvs(dataroot))
    return paths

def is_csv_file(filename):
    return any(filename.endswith(extension) for extension in CSV_EXTENSIONS)

def _get_paths_from_csvs(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    csvs = []
    for dirpath, _, fname in sorted(os.walk(path)):
        for fname in sorted(fname):
            if is_csv_file(fname):
                csv_path = os.path.join(dirpath, fname)
                csvs.append(csv_path)
    assert csvs, '{:s} has no valid csv file'.format(path)
    return csvs