from collections import OrderedDict
import os
import torch
import torch.nn as nn
from models.select_network import define_network

class ModelPlain():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models'] # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']     # training or not
        self.net = define_network(opt)
        self.net = self.net.to(self.device)

    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.net.train()                      # set training mode
        self.define_loss()                    # define_loss

    def load(self):
        load_path_model = self.opt['path']['pretrained_net']
        if load_path_model is not None:
            print('Loading model [{:s}] ...'.format(load_path_model))
            self.load_network(load_path_model, self.net, strict=self.opt['path']['strict_net'])

    def define_loss(self):

        if self.opt_train['lossfn_weight'] > 0:
            lossfn_type = self.opt_train['lossfn_type']
            if lossfn_type == 'CrossEntropy':
                self.lossfn = nn.CrossEntropyLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))
            self.lossfn_weight = self.opt_train['lossfn_weight']
        else:
            print('Do not use pixel loss.')
            self.lossfn = None
    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        load_network = torch.load(load_path)
        if param_key is not None:
            if param_key not in load_network and 'params' in load_network:
                param_key = 'params'
                print('Loading: params_ema does not exist, use params.')
            load_network = load_network[param_key]
        if strict:
            network.load_state_dict(load_network, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            state_dict = network.state_dict()
            for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict