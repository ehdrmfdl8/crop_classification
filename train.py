import argparse
import math
import os
from collections import OrderedDict

import torch
from data.dataset import Dataset
from utils import utils_logger
from utils import utils_option as option
from utils import utils_image
from torch.utils.data import DataLoader
from models.model_plain import ModelPlain
import logging

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
    opt['amp'] = parser.parse_args().amp

    scaler = torch.cuda.amp.GradScaler()

    # ----------------------------------------
    # configure logger
    # ----------------------------------------

    utils_image.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    logger_name = 'train'
    #utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

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
            train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
    '''
    # ----------------------------------------
    # Step--3 (initialize models)
    # ----------------------------------------
    '''
    model = ModelPlain(opt, scaler)

    model.init_train()

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    current_step = 0
    loss_plot, val_loss_plot = [], []
    metric_plot, val_metric_plot = [], []

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):
            current_step += 1
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data, need_label=True)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                idx = 0
                for test_data in test_loader:
                    idx += 1
                    model.is_train = False
                    model.feed_data(test_data)
                    model.test()

                    # -----------------------
                    # calculate F1_score
                    # -----------------------
                    logs = model.current_log()
                    val_loss = logs['loss']
                    f1_score = logs['score']

                    logger.info('{:->4d}--> {:>4.2f} | {:<4.2f}'.format(idx, val_loss, f1_score))

                val_loss_plot.append(val_loss)
                val_metric_plot.append(f1_score)
                model.is_train = True

            # -------------------------------
            # 6) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

if __name__ == '__main__':
    main()