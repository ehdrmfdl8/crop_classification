import numpy as np
import math
import random
import pandas as pd
import torch

import torch.utils.data as data
import utils.utils_image as util_image
import utils.utils_csv as util_csv
import utils.utils_json as util_json
import matplotlib.pyplot as plt
import cv2
import tqdm

# 제공된 sample data는 파프리카와 시설포도 2종류의 작물만 존재
sample_label_description = {
 '3_00_0': '파프리카_정상',
 '3_a9_1': '파프리카흰가루병_초기',
 '3_a9_2': '파프리카흰가루병_중기',
 '3_a9_3': '파프리카흰가루병_말기',
 '3_a10_1': '파프리카잘록병_초기',
 '3_a10_2': '파프리카잘록병_중기',
 '3_a10_3': '파프리카잘록병_말기',
 '3_b3_1': '칼슘결핍_초기',
 '3_b3_2': '칼슘결핍_중기',
 '3_b3_3': '칼슘결핍_말기',
 '3_b6_1': '다량원소결핍 (N)_초기',
 '3_b6_2': '다량원소결핍 (N)_중기',
 '3_b6_3': '다량원소결핍 (N)_말기',
 '3_b7_1': '다량원소결핍 (P)_초기',
 '3_b7_2': '다량원소결핍 (P)_중기',
 '3_b7_3': '다량원소결핍 (P)_말기',
 '3_b8_1': '다량원소결핍 (K)_초기',
 '3_b8_2': '다량원소결핍 (K)_중기',
 '3_b8_3': '다량원소결핍 (K)_말기',
 '6_00_0': '시설포도_정상',
 '6_a11_1': '시설포도탄저병_초기',
 '6_a11_2': '시설포도탄저병_중기',
 '6_a11_3': '시설포도탄저병_말기',
 '6_a12_1': '시설포도노균병_초기',
 '6_a12_2': '시설포도노균병_중기',
 '6_a12_3': '시설포도노균병_말기',
 '6_b4_1': '일소피해_초기',
 '6_b4_2': '일소피해_중기',
 '6_b4_3': '일소피해_말기',
 '6_b5_1': '축과병_초기',
 '6_b5_2': '축과병_중기',
 '6_b5_3': '축과병_말기',
}

crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
risk = {'1':'초기','2':'중기','3':'말기'}



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



def CSV_MinMax_Scaling(paths):

    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
            '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    csv_files = paths
    temp_csv = pd.read_csv(csv_files[0], encoding='utf-8')[csv_features]

    max_arr,min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # feature 별 최대값, 최솟값 계산
    for csv in csv_files[1:]:
        temp_csv = pd.read_csv(csv, encoding='utf-8')[csv_features]
        temp_csv = temp_csv.replace('-',np.nan).dropna()
        if len(temp_csv) == 0:
            continue
        temp_csv = temp_csv.astype(float)
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr,temp_max], axis=0)
        min_arr = np.min([min_arr,temp_min], axis=0)

    # feature 별 최대값,최솟값 dictionary 생성
    csv_feature_dict = {csv_features[i]:[min_arr[i],max_arr[i]] for i in range(len(csv_features))}
    return csv_feature_dict

class Dataset(data.Dataset):
    """
    # Get image dataset
    """
    def __init__(self, opt):
        super(Dataset, self).__init__()
        print('Get crop image.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        ## Data Augmentation 필요시 적용 (추가예정)

        self.paths = util_image.get_data_paths(opt['dataroot'])
        #csv_paths = util_csv.get_csv_paths(opt['dataroot'])
        csv_paths = [path+'/'+path.split('/')[-1]+'.csv' for path in self.paths]

        assert self.paths, 'Error : img_path is empty'
        self.csv_feature_dict = CSV_MinMax_Scaling(csv_paths)
        self.csv_feature_check = [0]*len(csv_paths)
        self.csv_features = [None]*len(csv_paths)
        self.max_len = 24 * 6
        self.label_encoder = define_Label(crop, disease, risk)

    def __getitem__(self, index):


        path = self.paths[index]
        file_name = path.split('/')[-1]

        # ------------------------------------
        # get csv
        # ------------------------------------
        if self.csv_feature_check[index] == 0:
            csv_path = f'{path}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling 정규화
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1] - self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[index] = csv_feature
            self.csv_feature_check[index] = 1
        else:
            csv_feature = self.csv_features[index]

        # ------------------------------------
        # get image
        # ------------------------------------
        img_path = f'{path}/{file_name}.jpg'
        img = util_image.imread_uint(img_path, self.n_channels)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = util_image.uint2tensor3(img)

        if self.opt['phase'] == 'train':
            json_path = f'{path}/{file_name}.json'
            json_file = util_json.parse(json_path)

            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'

            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32),
                'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32)
            }

    def __len__(self):
        return len(self.paths)

# if __name__ == '__main__':
#     root = 'E:/python/dacon/crop_classification/sample/sample_data'
#     dataset = Dataset(root)
#     sample_csv = pd.read_csv(dataset.csv_paths[3])
#     sample_image = util_image.imread_uint(dataset.img_paths[3])
#     sample_json = util_json.parse(dataset.json_paths[3])
#     util_image.imshow(sample_image)
#     print("ok")
#     # visualize bbox
#     plt.figure(figsize=(7,7))
#     points = sample_json['annotations']['bbox'][0]
#     part_points = sample_json['annotations']['part']
#
#     cv2.rectangle(
#         sample_image,
#         (int(points['x']), int(points['y'])), # left upper coord
#         (int((points['x']+points['w'])), int((points['y']+points['h']))), # right bottom coord
#         (0, 255, 0), # Green color
#         2 # thick of line
#     )
#     for part_point in part_points:
#         point = part_point
#         cv2.rectangle(
#             sample_image,
#             (int(point['x']), int(point['y'])),
#             (int((point['x'] + point['w'])), int((point['y'] + point['h']))),
#             (255, 0, 0),  # Red color
#             1 # thick of line
#         )
#     plt.imshow(sample_image)
#     plt.show()