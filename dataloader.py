import random
import os
from io import open

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
import numpy as np


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

    def get_assigned_data(self):
        return self.data[self.index]


class DataPartitioner(object):
    def __init__(self, data, batch_size, sizes=None, seed=1234, shuffle=True):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]
        self.data = data
        self.partitions = []
        self.bsz = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        if shuffle:
            rng = random.Random()
            rng.seed(seed)
            rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0: part_len])
            self.bsz.append(batch_size * frac)
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition]), self.bsz[partition]



def partition_dataset(dataset, partition_sizes, rank, batch_size, seed):
    if dataset == "wikitext2":
        rnn = True
    else:
        rnn = False

    if dataset == "mnist":
        dataset = datasets.FashionMNIST('./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
        testset = datasets.FashionMNIST('./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    elif dataset == "cifar10":
        dataset = datasets.CIFAR10('./data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                   ]))
        testset = datasets.CIFAR10('./data', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                   ]))
    elif dataset == "cifar100":
        dataset = datasets.CIFAR100('./data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                   ]))
        testset = datasets.CIFAR100('./data', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                   ]))
    elif dataset == "wikitext2":
        corpus = Corpus("rnn_data/wikitext-2")
        dataset = corpus.train
        testset = corpus.test

    if rnn:
        partition = DataPartitioner(dataset, batch_size, partition_sizes, shuffle=False)
        partition, bsz = partition.use(rank)
        train_set = batchify(partition.get_assigned_data(), bsz)
        eval_batch_size = 10
        val_set = batchify(testset, eval_batch_size)
    else:
        partition = DataPartitioner(dataset, batch_size, partition_sizes, seed=seed)
        partition, bsz = partition.use(rank)
        train_set = DataLoader(partition, batch_size=int(bsz), shuffle=True)
        val_set = DataLoader(testset, batch_size=int(bsz), shuffle=False)

    return train_set, val_set, bsz


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // int(bsz)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * int(bsz))
    # Evenly divide the data across the bsz batches.
    data = data.view(int(bsz), -1).t().contiguous()
    return data