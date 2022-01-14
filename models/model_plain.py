from collections import OrderedDict
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from models.select_network import define_network
from sklearn.metrics import f1_score

class ModelPlain():
    def __init__(self, opt, scaler):
        self.opt = opt
        self.save_dir = opt['path']['models'] # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']     # training or not
        self.net = define_network(opt)
        self.net = self.net.to(self.device)
        self.schedulers = []            # schedulers
        self.scaler = scaler            # Mixed precision
        self.amp = opt['amp']

    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.net.train()                      # set training mode
        self.define_loss()                    # define_loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

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


    def define_optimizer(self):
        optim_params = [] # optimizer parameter group
        for k, v in self.net.named_parameters():
            if v.requires_grad: # gradient가 필요한 요소만 업데이트
                optim_params.append(v)
        self.optimizer = Adam(optim_params, lr=self.opt_train['optimizer_lr'], weight_decay=0)
        del optim_params


    def load_optimizers(self):
        load_path_optimizer = self.opt['path']['pretrained_optimizer']
        if load_path_optimizer is not None and self.opt_train['optimizer_reuse']:
            print('Loading optimizer [{:s}] ...'.format(load_path_optimizer))
            self.load_optimizer(load_path_optimizer, self.optimizer)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        scheduler_type = self.opt_train['scheduler_type']
        if scheduler_type == "MultiStepLR":
            self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer,
                                                            self.opt_train['scheduler_milestones'],
                                                            self.opt_train['scheduler_gamma']
                                                            ))
        elif scheduler_type == "CyclicLR":
            self.schedulers.append(lr_scheduler.CyclicLR(self.optimizer,
                                                         base_lr=1e-5,
                                                         max_lr=2e-4,
                                                         step_size_up=5,
                                                         step_size_down=100,
                                                         mode='exp_range',
                                                         gamma=0.9995))
        else:
            print('error : set scheduler_type')

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

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def feed_data(self, data, need_label=False):
        self.img = data['img'].to(self.device)
        self.csv_feature = data['csv_feature'].to(self.device)
        if need_label:
            self.label = data['label'].to(self.device)

    # ----------------------------------------
    # feed img and csv to net
    # ----------------------------------------
    def net_forward(self):
        self.out = self.net(self.img, self.csv_feature)

    def accuracy_function(self, real, pred):
        score = f1_score(real, pred, average='macro')
        return score

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        if self.amp:
            with torch.cuda.amp.autocast():
                self.net_forward()

                loss_total = 0

                if self.opt_train['lossfn_weight'] > 0:
                    loss = self.lossfn_weight * self.lossfn(self.out, self.label)
                    loss_total += loss  # 1) loss
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.net_forward()

            loss_total = 0

            if self.opt_train['lossfn_weight'] > 0:
                loss = self.lossfn_weight * self.lossfn(self.out, self.label)
                loss_total += loss  # 1) loss

            # BBOX에 관한 loss 추가 예정

            loss_total.backward()
            self.optimizer.step()

        if self.opt_train['lossfn_weight'] > 0:
            self.log_dict['loss'] = loss.item()


    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.net, 'net', iter_label)
        if self.opt_train['optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.optimizer, 'optimizer', iter_label)

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label, param_key='params'):
        save_filename = '{}_{}.pth'.format(network_label, iter_label)
        save_path = os.path.join(save_dir, save_filename)
        network = network if isinstance(network, list) else [network]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(network) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(network, param_key):
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict
        torch.save(save_dict, save_path)

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.net.eval()
        with torch.no_grad():
            if self.amp:
                with torch.cuda.amp.autocast():
                    self.net_forward()
            else:
                self.net_forward()
        self.net.train()

