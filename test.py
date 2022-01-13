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

def define_Label(crop, disease, risk):
    label_description = {}
    for key, value in disease.items():
        label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
        for disease_code in value:
            for risk_code in risk:
                label = f'{key}_{disease_code}_{risk_code}' # key 값 정의 (코드)
                label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}' # value 값 정의 (명칭)
    label_encoder = {key: idx for idx, key in enumerate(label_description)}  # index를 추출
    # label_decoder = {val: key for key, val in label_encoder.items()}  # word를 추출
    return label_encoder
def define_Dictionary(crop):
    dictionary = {}
    for key, value in crop.items():
        dictionary[value] = []
    return dictionary
crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}

area = {}
risk = {'1':'초기','2':'중기','3':'말기'}

if __name__ == '__main__':

    path ='E:/python/dacon/crop_classification/data'

    paths = get_json_paths(path)
    crop_dict = define_Dictionary(crop)
    for path in paths:
        file = parse(path)
        crop_dict[crop[str(file['annotations']['crop'])]] += [file['description']['image']]
