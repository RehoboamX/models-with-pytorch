import torchvision
from torch.utils import data
from torchvision import transforms

normlize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # RGB三通道标准化


def load_data_cifar10(batch_size):
    trans_train = [transforms.Resize((224, 224)),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normlize]
    trans_train = transforms.Compose(trans_train)
    trans_val = [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 normlize]
    trans_val = transforms.Compose(trans_val)
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=trans_train, download=True)
    val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=trans_val, download=True)
    return (data.DataLoader(train_dataset, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(val_dataset, batch_size, shuffle=False,
                            num_workers=0))

def load_data_cifar100(batch_size):
    trans_train = [transforms.Resize((224, 224)),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normlize]
    trans_train = transforms.Compose(trans_train)
    trans_val = [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 normlize]
    trans_val = transforms.Compose(trans_val)
    train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, transform=trans_train, download=True)
    val_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, transform=trans_val, download=True)
    return (data.DataLoader(train_dataset, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(val_dataset, batch_size, shuffle=False,
                            num_workers=0))