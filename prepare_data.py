from torchvision import datasets


def prepare_data():
    datasets.FashionMNIST('./data', train=True, download=True)
    datasets.CIFAR10('./data', train=True, download=True)


if __name__ == "__main__":
    prepare_data()