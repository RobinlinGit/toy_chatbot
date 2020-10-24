#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2020/10/08 23:22:18
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   model for chatbot
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer


def make_pad_mask(src, pad):
    """produce pad mask for src or trg
    """
    # src = [src_len, batch_size]
    src_mask = src.permute(1, 0) == pad
    # src_mask [batch size, src len]
    return src_mask


def make_mask(src):
    # mask [src len, src len]
    sz = src.size(0)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        temp = position * div_term
        self.pe[:, 0::2] = torch.sin(temp)
        self.pe[:, 1::2] = torch.cos(temp)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device,
                 dropout=0.5, src_pad_idx=0, trg_pad_idx=0):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.emb = nn.Embedding(ntoken, ninp)
        self.fc_out = nn.Linear(ninp, ntoken)
        self.transformer = Transformer(ninp, nhead, nlayers, nlayers, dropout=dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    
    def forward(self, src, trg):
        # src [src len, batch size]
        src_mask = make_mask(src).to(self.device)
        trg_mask = make_mask(trg).to(self.device)
        src_pad_mask = make_pad_mask(src, self.src_pad_idx).to(self.device)
        trg_pad_mask = make_pad_mask(trg, self.trg_pad_idx).to(self.device)
        src = self.emb(src)
        trg = self.emb(trg)
        
        output = self.transformer(src, trg, src_mask=src_mask, tgt_mask=trg_mask,
                                  src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=trg_pad_mask)
        # print(f"output {output.size()}")
        return self.fc_out(output)


def greedy_decode(model, src, device, SOS, PAD, EOS, max_len=50):
    """greedy decode for seq2seq.

    Args:
        model (nn.Module): seq2seq model;
        src (list): list of index.
        SOS (int): start of sentence index.
        PAD (int): pad index.
        EOS (int): end of sentence index.
        device (torch.device): device the model run on.
        max_len (int): max length of pred seq.
    Returns:
        trg (list [int]): seq of target index.
    """
    src = torch.LongTensor(src).permute(1, 0).to(device)
    src_mask = make_mask(src)
    src_pad_mask = make_pad_mask(src, PAD)
    enc_src = model.encoder(src, src_mask, src_pad_mask)

    trg = [SOS]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg).permute(1, 0)
        trg_mask = make_mask(trg)
        output = model.decoder(trg_tensor, trg_mask=trg_mask)
        # output [trg len, batch size(1), ntokens]
        pred_token = output.argmax(2)[:, -1].item()
        trg.append(pred_token)
        if pred_token == EOS:
            break
    return trg


