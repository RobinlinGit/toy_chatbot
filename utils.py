#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/10/07 11:16:21
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   chatbot util functions
'''

import json
import torch
import torch.nn.utils.rnn as rnn_utils
import random
from tqdm import tqdm
import jieba

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def file_check(filetype):
    def decorator(func):
        def wrapper(*args, **kwargs):
            assert isinstance(args[0], str), f"{func.__name__} args[0] should be str"
            assert args[0].endswith(filetype), f"{func.__name__} args[0] shoule be {filetype} file"
            return func(*args, **kwargs)
        return wrapper
    return decorator


@file_check('tsv')
def load_lines(filename):
    """load file to a list of (src, trg) pair

    Args:
        filename (str): tsv file path, only accept tsv file
    Returns:
        lines (list [tuple (str, str)]): list of (src, trg)
    """
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            lines.append(line.split('\t')[:2])
    return lines

def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def parse_sentence(sentence: str):
    """parse sentence to a list of word, support chinese with english mixed.

    Args:
        sentence (str);
    Returns:
        word_list (list [str]);
    """
    word_list = list(jieba.cut(sentence))
    word_list = filter(lambda x: not (x.isspace() or len(x) == 0), word_list)
    words = []
    for w in word_list:
        if is_chinese(w[0]):
            words += w.split()
        else:
            words.append(w)
    return words

            
class DictMaker(object):
    """generate a dict from text data, and transfer text to index array.
    see https://pytorch.org/tutorials/beginner/chatbot_tutorial.html for more.

    Attributes:
        name (str): name of the corpus.
    """

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
        self.word2index = {}
        self.word2count = {}

    def add_sentence(self, sentence: str):
        for word in parse_sentence(sentence):
            
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.index2word[self.num_words] = word
                self.word2count[word] = 1
                self.num_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):
        """trim some words with frequency below min_count.

        Args:
            min_count (int): min frequency.
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        print(f"keep words: {len(keep_words)} / {len(self.word2count)}")
        self.num_words = 3
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}

        for word in keep_words:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

    def __str__(self):
        s = (
            "Class DictMaker\n"
            f"corpus: {self.name}\n"
            f"word: {len(self.word2count)}\n"
            f"trimmed: {self.trimmed}"
        )
        return s

    def dump(self, filename):
        """dump the dict to file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                {
                    "word2index": self.word2index,
                    "index2word": self.index2word,
                    "word2count": self.word2count
                }
            ))
    
    def load(self, filename):
        """load the dict from file.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.word2count = data['word2count']
        self.word2index = data['word2index']
        self.index2word = data['index2word']
        self.num_words = len(self.word2count)

    def indexs2sentence(self, index_list):
        """transfer index list to sentence.

        Args:
            index_list (list [int]): index list.
        Returns:
            sentence (str)
        Raises:
            AssertError: word not in dict.
        """
        sentence = ''.join([self.index2word[x] for x in index_list])
        return sentence

    def sentence2index(self, sentence):
        sentence = parse_sentence(sentence)
        return [self.word2index[x] for x in sentence]


class TextPairDataset(torch.utils.data.Dataset):
    """Dataset for text data (in_text, out_text)

    default SOS and EOS are 1 and 2;

    Attributes:
        seq_list (list [tuple]): list of text data (in_text, out_text);
        voc (DictMaker): vocabulary for data;
        lines (list [(str, str)]): list of origin str, str file;
    """

    def __init__(self, filename, read_func=load_lines, max_len=40):
        """load lines in file and prepare data.

        Args:
            filename (str): data file path.
            SOS (int): SOS
        """
        self.lines = read_func(filename)
        self.voc = DictMaker(filename)
        self.read_func = read_func
        self.max_len = max_len
        self.seq_list = []
        self.build_voc()
    
    def build_voc(self):
        for line in tqdm(self.lines, desc="build voc......"):
            self.voc.add_sentence(line[0])
            self.voc.add_sentence(line[1])
        self.seq_list = [
            (self.voc.sentence2index(x[0]), self.voc.sentence2index(x[1]))
            for x in tqdm(self.lines, desc="sentence to index")
        ]
        self.seq_list = list(filter(
            lambda x: len(x[0]) <= self.max_len and len(x[1]) <= self.max_len,
            self.seq_list)
        )
        print(f"voc size: {self.voc.num_words}")
        print(f"data size: {len(self.seq_list)}")
        
    def __len__(self):
        return len(self.seq_list)
    
    def __getitem__(self, index):
        text_in, text_out = self.seq_list[index]
        return (
            torch.LongTensor([SOS_token] + text_in + [EOS_token]),
            torch.LongTensor([SOS_token] + text_out + [EOS_token])
        )


def collate_fn_for_text_pair(data):
    """
    Args:
        data (list)
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = torch.IntTensor([len(sq[0]) for sq in data])
    data_in = rnn_utils.pad_sequence([x[0] for x in data], padding_value=PAD_token)
    data_out = rnn_utils.pad_sequence([x[1] for x in data], padding_value=PAD_token)
    return data_in, data_out, data_length