#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2020/10/22 15:06:08
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   config for model
'''
import torch.nn as nn
import torch.optim as optim


class Configs:
    ninp = 128
    nhead = 4
    nhid = 256
    nlayers = 2
    dropout = 0.5
    # below are train config
    batch_size = 64
    epochs = 20
    CLIP = 1
    valid_ratio = 0.1
    lr = 0.001
