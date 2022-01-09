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
label_description = {
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

label_encoder = {key:idx for idx, key in enumerate(label_description)}
label_decoder = {val:key for key, val in label_encoder.items()}

def CSV_MinMax_Scaling(paths):

    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
            '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    csv_files = paths
    temp_csv = pd.read_csv(csv_files[0])[csv_features]
    max_arr,min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # feature 별 최대값, 최솟값 계산
    for csv in tqdm(csv_files[1:]):
        temp_csv = pd.read_csv(csv)[csv_features]
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
        self.is_train = opt['is_train']
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        ## Data Augmentation 필요시 적용 (추가예정)

        self.img_paths = util_image.get_image_paths(opt['dataroot'])
        self.csv_paths = util_csv.get_csv_paths(opt['dataroot'])
        self.json_paths = util_json.get_json_paths(opt['dataroot'])
        assert self.img_paths, 'Error : img_path is empty'
        assert self.csv_paths, 'Error : csv_path is empty'
        assert self.json_paths, 'Error : json_path is empty'
        self.csv_feature_dict = CSV_MinMax_Scaling(self.csv_paths)
        self.csv_feature_check = [0]*len(self.csv_paths)
        self.csv_features = [None]*len(self.csv_paths)
        self.max_len = -1 * 24 * 6
        self.label_encoder = label_encoder

    def __getitem__(self, index):

        # ------------------------------------
        # get image
        # ------------------------------------
        path = self.img_paths[index]
        # file_name = path.split('/')[-1]
        if self.csv_feature_check[index] == 0:
            df = pd.read_csv(self.csv_paths[index])

            # MinMax scaling 정규화
            for col in self.csv_feature_dict.keys():
                df[col] = df[col] - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1] - self.csv_feature_dict[col][0])

            # transpose to sequential data
            csv_feature = df[self.csv_feature_dict.keys()].to_numpy()[self.max_len:].T
            self.csv_features[index] = csv_feature
            self.csv_feature_check[index] = 1
        else:
            csv_feature = self.csv_features[index]

        img = util_image.imread_uint(path, self.n_channels)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = util_image.uint2tensor3(img)

        if self.is_train:
            json_file = util_json.parse(self.json_paths[index])
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
        return len(self.img_paths)

if __name__ == '__main__':
    root = 'E:/python/crop_classification/sample/sample_data'
    dataset = Dataset(root)
    sample_csv = pd.read_csv(dataset.csv_paths[3])
    sample_image = util_image.imread_uint(dataset.img_paths[3])
    sample_json = util_json.parse(dataset.json_paths[3])
    util_image.imshow(sample_image)
    print("ok")
    # visualize bbox
    plt.figure(figsize=(7,7))
    points = sample_json['annotations']['bbox'][0]
    part_points = sample_json['annotations']['part']

    cv2.rectangle(
        sample_image,
        (int(points['x']), int(points['y'])), # left upper coord
        (int((points['x']+points['w'])), int((points['y']+points['h']))), # right bottom coord
        (0, 255, 0), # Green color
        2 # thick of line
    )
    for part_point in part_points:
        point = part_point
        cv2.rectangle(
            sample_image,
            (int(point['x']), int(point['y'])),
            (int((point['x'] + point['w'])), int((point['y'] + point['h']))),
            (255, 0, 0),  # Red color
            1 # thick of line
        )
    plt.imshow(sample_image)
    plt.show()