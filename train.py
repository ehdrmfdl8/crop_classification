import argparse
import math
from collections import OrderedDict

import torch
from data.dataset import Dataset
from utils import utils_option as option
from torch.utils.data import DataLoader

def main(option_path='options/train_resnet_lstm.yaml'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=option_path, help='Path to option JSON file.')
    parser.add_argument('--amp', default=True)
    parser.add_argument('--resume', default=True)
    opt = option.parse(parser.parse_args().opt, is_train=True)

    print(opt)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''
    # 1) create dataset
    # 2) create dataloader for train and test
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            train_dataloader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)
        elif phase == 'test':
            test_set = Dataset(dataset_opt)
    '''
    # ----------------------------------------
    # Step--3 (initialize models)
    # ----------------------------------------
    '''
    # models = define_Model(opt)
    #
    # models.init_train()
if __name__ == '__main__':
    main()