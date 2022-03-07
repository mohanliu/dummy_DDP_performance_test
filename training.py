import os
import time

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
LOCAL_WORLD_SIZE = int(os.environ["AZ_BATCHAI_GPU_COUNT_NEED"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

LEARNING_RATE = 5e-5
SCALED_LEARNING_RATE = LEARNING_RATE * WORLD_SIZE


class InMemoryDataset(torch.utils.data.Dataset):
    """PyTorch dataset for the images in MS COCO."""

    def __init__(self, dataset, length):
        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224]),
            ]
        )

        self._len = length
        self._items = []
        for i in range(10):
            item = dataset.__getitem__(i)
            item = (
                self._transform(item[0].convert("RGB")),
                torch.as_tensor(item[1], dtype=torch.float),
            )
            self._items.append(item)

    def __getitem__(self, index):
        return self._items[index % len(self._items)]

    def __len__(self):
        return self._len


def _get_device():
    if torch.cuda.is_available():
        return torch.device(LOCAL_RANK)
    return torch.device("cpu")


device = _get_device()


def init_optimizer(model):
    return torch.optim.SGD(
        model.parameters(),
        lr=SCALED_LEARNING_RATE,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )


def init_model():
    model = torchvision.models.resnet152(pretrained=True)
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, 1)
    model = model.to(device)
    model = DDP(model)
    return model


def train_epoch(model, train_data_loader, optimizer):
    criterion = nn.BCEWithLogitsLoss()
    for images, targets in tqdm(train_data_loader):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.squeeze(), targets.squeeze())

        loss.backward()
        optimizer.step()


if __name__ == "__main__":

    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    dist.init_process_group("nccl", rank=RANK, world_size=WORLD_SIZE)

    train_dataset = InMemoryDataset(
        dataset=torchvision.datasets.MNIST(root=str(RANK), download=True), length=100000
    )
    model = init_model()
    model.train()
    optimizer = init_optimizer(model)

    num_workers = os.cpu_count() // LOCAL_WORLD_SIZE
    print(f"Number of PyTorch data loader workers: {num_workers}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=WORLD_SIZE, rank=RANK
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    for epoch in range(10):

        print(f"Starting epoch {epoch}")

        train_sampler.set_epoch(epoch)

        start = time.time()
        train_epoch(model, train_data_loader, optimizer)
        if RANK == 0:
            print(f"Epoch training time: {time.time() - start}")

        model.eval()

    print("Done")
