import os
import math
import random
import numpy as np
import torch
import json
import yaml
from collections import OrderedDict
JSON_EXTENSIONS = ['.json', '.JSON']

def get_json_paths(dataroot):
    paths = None # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_jsons(dataroot))
    return paths

def is_json_file(filename):
    return any(filename.endswith(extension) for extension in JSON_EXTENSIONS)

def _get_paths_from_jsons(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    jsons = []
    for dirpath, _, fname in sorted(os.walk(path)):
        for fname in sorted(fname):
            if is_json_file(fname):
                json_path = os.path.join(dirpath, fname)
                jsons.append(json_path)
    assert jsons, '{:s} has no valid csv file'.format(path)
    return jsons

def parse(path):
    json_str = ''
    with open(path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    return opt