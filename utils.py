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
        for word in sentence:
            if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] += 1
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

    def __str__(self):
        print("Class DictMaker")
        print(f"corpus: {self.name}")
        print(f"word: {len(self.word2count)}")
        print(f"trimmed: {self.trimmed}")

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
        for index in index_list:
            if index > self.num_words:
                raise AssertionError("word not in dict")

        sentence = ''.join([self.index2word[x] for x in index_list])
        return sentence

    