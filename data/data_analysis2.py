'''
crop_area_disease_risk : [파일이름, 최고_최저 딕셔너리, Disease box 차지 비율]
형식으로 되어 있음.
------------------------------------------------------
추가:   data_ndarray = np.append(추가하고 싶은 데이터)
------------------------------------------------------
'''

from data.dataset import *
from utils.utils_json import *

from utils.utils_csv import _get_paths_from_csvs
from utils.utils_json import _get_paths_from_jsons

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

if __name__ == '__main__':
    data_dict = {}
    path = 'data/train'
    paths_json = _get_paths_from_jsons(path)
    paths_csv = _get_paths_from_csvs(path)

    for path in paths_json:
        basename = os.path.basename(path)
        file_name = os.path.splitext(basename)[0]

        file = parse(path)
        crop_type = crop[str(file['annotations']['crop'])]
        disease_num = disease[str(file['annotations']['crop'])][str(file['annotations']['disease'])]
        area_num = area[str(file['annotations']['area'])]
        disease_type = f'{area_num}_{disease_num}'
        risk_type = risk[str(file['annotations']['risk'])]

        points = file['annotations']['bbox'][0]
        points_w = points['w']
        points_h = points['h']
        part_points = file['annotations']['part']

        label = f'{crop_type}_{disease_type}_{risk_type}'

        basename = os.path.basename(path)
        file_name = os.path.splitext(basename)[0]
        data_ndarray = np.array(file_name)  # 파일 이름 array에 저장

        csv_feature_dict = CSV_MinMax_Scaling(path)
        data_ndarray = np.append(data_ndarray, csv_feature_dict)  # 9개 데이터 최고, 최저 값 저장

        if len(part_points) > 0:
            size_sum = 0
            for part_point in part_points:
                part_w = part_point['w']
                part_h = part_point['h']
                size = (part_w / points_w) * (part_h / points_h)
                size_sum += size

            data_ndarray = np.append(data_ndarray, size_sum)  # 작은 박스 넓이 저장

        if label not in data_dict:
            data_dict[label] = []
        data_dict[label].append(data_ndarray)
