
'''
4중 딕셔너리
crop_dict{crop_type: {area_type: {disease_type: {risk_type: [파일이름, 최고_최저 딕셔너리, Disease box 차지 비율] } } } }
형식으로 되어 있음.
----------------------------------------------------------------
호출:   crop_dict[crop_type][area_type][disease_type][risk_type]
----------------------------------------------------------------
추가:   data_ndarray = np.append(추가하고 싶은 데이터)
----------------------------------------------------------------
'''

import copy

from utils.utils_csv import _get_paths_from_csvs
from utils.utils_json import _get_paths_from_jsons
from utils.utils_json import *

from data.dataset import *

crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
area = {'1': '열매', '2': '꽃', '3': '잎', '4': '가지', '5': '줄기', '6': '뿌리', '7': '해충'}

disease = {'1': {'00': '정상', 'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '2': {'00': '정상', 'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)',
                 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
           '3': {'00': '정상', 'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '4': {'00': '정상', 'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '5': {'00': '정상', 'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '6': {'00': '정상', 'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}}

risk = {'0': '정상', '1': '초기', '2': '중기', '3': '말기'}

def define_Dictionary(): # 4중 딕셔너리에 빈 list 만들기
    crop_dict = {}
    area_dict = {}
    disease_dict = {}
    risk_dict = {}
    for v in risk.values(): # ex. 정상, 초기, 중기, 말기
        if v != '정상': # 초기, 중기, 말기
            risk_dict[v] = []

    for k1, v1 in disease.items(): # ex. 1(딸기)
        temp_dict = {}
        for k2, v2 in v1.items(): # ex. 00 : 정상
            if k2 == '00':
                temp_dict[v2] = {'정상': []}
            else:
                temp_dict[v2] = copy.deepcopy(risk_dict) # {'초기':[], '중기':[], '말기':[]}

        disease_dict[k1] = copy.deepcopy(temp_dict)

    for k3, v3 in crop.items(): # ex. 1(딸기)
        for v4 in area.values(): # ex. 1(열매)
            area_dict[v4] = copy.deepcopy(disease_dict[k3])
        crop_dict[v3] = copy.deepcopy(area_dict)
    return crop_dict

if __name__ == '__main__':
    path = 'train/train'
    paths_json = _get_paths_from_jsons(path)
    paths_csv = _get_paths_from_csvs(path)

    crop_dict = define_Dictionary()

    for path in paths_json:
        basename = os.path.basename(path)
        file_name = os.path.splitext(basename)[0] # 파일 이름 추출 -> ex. 10027
        file = parse(path)
        crop_type = crop[str(file['annotations']['crop'])]
        area_type = area[str(file['annotations']['area'])]
        disease_type = disease[str(file['annotations']['crop'])][str(file['annotations']['disease'])]
        risk_type = risk[str(file['annotations']['risk'])]

        points = file['annotations']['bbox'][0] # bbox 크기 추출
        points_w = points['w']
        points_h = points['h']
        part_points = file['annotations']['part'] # 작은 박스 크기 추출

        data_ndarray = np.array(file_name) # 파일 이름 array에 저장

        csv_feature_dict = CSV_MinMax_Scaling(path)
        data_ndarray = np.append(data_ndarray, csv_feature_dict) # 9개 데이터 최고, 최저 값 저장

        if len(part_points) > 0:
            size_sum = 0
            for part_point in part_points:
                part_w = part_point['w']
                part_h = part_point['h']
                size = (part_w / points_w) * (part_h / points_h)
                size_sum += size

            data_ndarray = np.append(data_ndarray, size_sum) # 작은 박스 넓이 저장

        crop_dict[crop_type][area_type][disease_type][risk_type].append(data_ndarray)