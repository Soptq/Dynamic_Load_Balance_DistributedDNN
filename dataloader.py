import random

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, batch_size, sizes=None, seed=1234):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]
        self.data = data
        self.partitions = []
        self.bsz = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0: part_len])
            self.bsz.append(batch_size * frac)
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition]), self.bsz[partition]


def partition_dataset(dataset, partition_sizes, rank, batch_size, seed):
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
    partition = DataPartitioner(dataset, batch_size, partition_sizes, seed=seed)
    partition, bsz = partition.use(rank)
    train_set = DataLoader(partition, batch_size=int(bsz), shuffle=True)
    val_set = DataLoader(testset, batch_size=int(bsz), shuffle=False)

    return train_set, val_set, bsz