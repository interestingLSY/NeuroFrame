import os, random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import tqdm

N_EPOCHS = 128
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 65536

LEARNING_RATE = 2e-2

RANDOM_SEED = 0
DISABLE_CUDNN = False

OPTIMZER_STEP = 1
OPTIMIZER_GAMMA = 0.995

DEVICE = torch.device("cuda")
DTYPE = torch.float32

torch.set_default_dtype(DTYPE)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.fc3 = nn.Linear(128, 10, bias=False)

    def forward(self, x: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        loss = F.cross_entropy(F.softmax(x, dim=1), ground_truth, reduction="sum")
        return x, loss
        

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if DISABLE_CUDNN:
        torch.backends.cudnn.enabled = False
    
    # Load dataset
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ]
    )
    def move_dataset_to_gpu(dataset):
        dataset.data = dataset.data.to(DEVICE)
        dataset.targets = dataset.targets.to(DEVICE)
        return dataset
    train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('/data/pytorch-dataset', train=True, download=True,
            transform=dataset_transform),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/pytorch-dataset', train=False, download=True,
            transform=dataset_transform),
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=4
    )

    # Create model
    net = Net().to(DEVICE)

    # Create optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=OPTIMZER_STEP, gamma=OPTIMIZER_GAMMA)

    # Move all dataset to GPU
    train_dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for data, target in train_loader:
        train_dataset.append((data.to(DEVICE), target.to(DEVICE)))
    test_dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for data, target in test_loader:
        test_dataset.append((data.to(DEVICE), target.to(DEVICE)))


    for epoch_idx in tqdm.tqdm(range(N_EPOCHS)):
        # Feed our network with every batch from the training set
        train_loss = 0      
        for (data, target) in train_dataset:
            optimizer.zero_grad()	# Clear gradientsssh 
            _, loss = net(data, target)	# Forward pass
            train_loss += loss.item()
            loss.backward()		# Backward pass (accumulate gradients)
            optimizer.step()	# Update weights
        train_loss /= len(train_loader.dataset)
        
        # Evaluate our network on the test set
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataset:
                pred_output, loss = net(data, target)
                test_loss += loss.item()
                pred = pred_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        print(f"Epoch {epoch_idx+1}/{N_EPOCHS} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")

        scheduler.step()	# Update learning rate