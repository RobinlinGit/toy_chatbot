#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/10/22 00:28:43
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   train model for chatbot
'''
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
import time
from models import TransformerModel
from utils import DictMaker, TextPairDataset, collate_fn_for_text_pair
from config import Configs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("filename", help="text pair data file", type=str)
    parser.add_argument("vocpath", help="voc save path", type=str,
                        default="voc.json", metavar="vocpath")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")
    parser.add_argument("--model-name", default="transformer", type=str)
    return parser.parse_args()


def train(model, iterator, optimizer, clip, criterion, device):
    model.train()

    epoch_loss = 0

    for batch in tqdm(iterator, desc="train"):
        src, trg, _ = batch
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()

        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg.contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        del trg
        del src
        del output
        del loss
        if device.type == "cuda":
            torch.cuda.empty_cache()


    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="test"):
            src, trg, _ = batch
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)

            loss = criterion(output, trg)
            loss.backward()
            
            epoch_loss += loss.item()
            del output
            del src
            del trg
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    args = parse_args()
    N = args.log_interval
    model_name = args.model_name
    N_EPOCHS = Configs.epochs
    CLIP = Configs.CLIP
    BATCH_SIZE = Configs.batch_size
    lr = Configs.lr
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    # print configs 
    for k, v in Configs.__dict__.items():
        print(f"{k}: {v}")
    

    train_loss = 0
    valid_loss = 0
    best_valid_loss = float('inf')

    dataset = TextPairDataset(args.filename)
    valid_len = int(Configs.valid_ratio * len(dataset))
    train_len = len(dataset) - valid_len
    train_set, test_set = random_split(dataset, [train_len, valid_len])
    train_iter = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn_for_text_pair)
    test_iter = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn_for_text_pair)
    
    ntokens = dataset.voc.num_words
    ninp = Configs.ninp
    nhid = Configs.nhid
    nhead = Configs.nhead
    nlayers = Configs.nlayers
    dropout = Configs.dropout
    dataset.voc.dump(args.vocpath)

    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, device, dropout).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = 0
    test_loss = 0

    for epoch in range(N_EPOCHS):
        if epoch % N == 0:
            start = time.time()
        train_loss += train(model, train_iter, optimizer, CLIP, criterion, device)

        if (epoch + 1) % N == 0:
            test_loss = evaluate(model, test_iter, criterion, device)
            end_time = time.time()
        
            epoch_mins, epoch_secs = epoch_time(start, end_time)
            
            if test_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'{model_name}.pt')
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss / N:.3f}')
            print(f'\t Val. Loss: {test_loss:.3f}')
            train_loss = 0

            
if __name__ == "__main__":
    main()