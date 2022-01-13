import argparse
import math
import os
from collections import OrderedDict
import warnings

import torch
from data.dataset import Dataset
from utils import utils_logger
from utils import utils_option as option
from utils import utils_image
from torch.utils.data import DataLoader
from models.model_plain import ModelPlain
import logging
warnings.filterwarnings(action='ignore')

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
    opt['resume'] = parser.parse_args().resume

    scaler = torch.cuda.amp.GradScaler()



    utils_image.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    if opt['resume'] == True:
        init_iter, init_path = option.find_last_checkpoint(opt['path']['models'], net_type='net')
        init_iter_optimizer, init_path_optimizer = option.find_last_checkpoint(opt['path']['models'],
                                                                                 net_type='optimizer')
    else:
        init_iter = 0
        init_path = opt['path']['pretrained_netG']
        init_iter_optimizer = 0
        init_path_optimizer = None

    opt['path']['pretrained_net'] = init_path
    opt['path']['pretrained_optimizer'] = init_path_optimizer
    current_step = max(init_iter, init_iter_optimizer)

    option.save(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
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
    best = 0
    for epoch in range(1000):  # keep running
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0
        train_pred = []
        train_label = []
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
            model.optimize_parameters() # forward & backward

            # -------------------------------
            # 4) training information
            # -------------------------------

            logs = model.current_log()  # such as loss, score

            total_loss += logs['loss'] / len(train_loader)
            train_pred += model.out.argmax(1).detach().cpu().numpy().tolist()
            train_label += model.label.detach().cpu().numpy().tolist()

        train_f1 = model.accuracy_function(train_label, train_pred)

        val_pred = []
        val_label = []
        # -------------------------------
        # 5) testing
        # -------------------------------
        if epoch % opt['train']['checkpoint_test'] == 0:
            for test_data in test_loader:

                model.is_train = False
                model.feed_data(test_data, need_label=True)
                model.test()
                val_pred += model.out.argmax(1).detach().cpu().numpy().tolist()
                val_label += model.label.detach().cpu().numpy().tolist()
            val_f1 = model.accuracy_function(val_label, val_pred)
            model.is_train = True

        # -------------------------------
        # 6) save model
        # -------------------------------
        if val_f1 >= best:
            logger.info('Saving the model.')
            best = val_f1
            model.save(epoch)

        message = f'epoch: {epoch+1}/{1000} train loss : {total_loss:.5f}  f1:{train_f1:.5f}  valid f1:{val_f1:.5f}  best f1:{best:.5f}'
        logger.info(message)

if __name__ == '__main__':
    main()