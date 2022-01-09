import functools
import torch
from torch.nn import init
import torch.nn as nn
from timm.models.layers import trunc_normal_


# --------------------------------------------
# select network
# --------------------------------------------
def define_network(opt):
    opt_net = opt['net']
    net_type = opt_net['net_type']

    if net_type == 'resnet_lstm':  # resnet_lstm
        from models.network_ResNet50_LSTM import CNN2RNN as net
        net = net(
            max_len=opt_net['max_len'],
            embedding_dim=opt_net['embedding_dim'],
            num_features=opt_net['num_features'],
            class_n=opt_net['class_n'],
            rate=opt_net['rate']
        )

    else:
        raise NotImplementedError('net [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(net,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return net


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            elif init_type == 'trunc_normal':
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))
        elif classname.find('LayerNorm') != -1:
            if init_bn_type == 'constant':
                if isinstance(m, nn.LayerNorm):
                    init.constant_(m.bias, 0.0)
                    init.constant_(m.weight, 1.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)
