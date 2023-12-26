import os, random, time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import tqdm

N_EPOCHS = 128
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 65536

LEARNING_RATE = 5e-4

RANDOM_SEED = 0
DISABLE_CUDNN = False

OPTIMZER_STEP = 1
OPTIMIZER_GAMMA = 0.99

DEVICE = torch.device("cuda")
DTYPE = torch.float32

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.fc1= nn.Linear(2*7*7, 10, bias=False)
        self.fc1.weight.data = np.random.uniform(-0.01, 0.01, self.fc1.weight.data.shape).astype(np.float32)

    def forward(self, x: torch.Tensor, ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.pool1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 2*7*7)
        x = self.fc1(x)
        loss = F.cross_entropy_loss(F.softmax(x, dim=1), ground_truth, reduction="sum")
        return x, loss
        

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if DISABLE_CUDNN:
        torch.backends.cudnn.enabled = False
    
    # Load dataset
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(
        #     (0.1307,), (0.3081,))
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
    optimizer = torch.optim.SGD(list(net.parameters()), lr=LEARNING_RATE)

    # Move all dataset to GPU
    train_dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for data, target in train_loader:
        train_dataset.append((data.to(DEVICE), target.to(DEVICE)))
    test_dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for data, target in test_loader:
        test_dataset.append((data.to(DEVICE), target.to(DEVICE)))


    history_train_losses = []
    history_test_losses = []
    start_time = time.time()
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
                # correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        
        history_train_losses.append(train_loss)
        history_test_losses.append(test_loss)
        print(f"Epoch {epoch_idx+1}/{N_EPOCHS} | Train loss: {train_loss:.6f} | Test loss: {test_loss:.6f} | Accuracy: {accuracy*100:.2f}%")

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} s")
    
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(history_train_losses, label="train")
    ax0.plot(history_test_losses, label="test")
    ax0.set_ylabel("loss")
    ax0.set_xlabel("iteration")
    ax0.set_title("loss vs iteration")
    ax0.legend()

    ax1.plot(np.log(history_train_losses), label="train")
    ax1.plot(np.log(history_test_losses), label="test")
    ax1.set_ylabel("loss (log)")
    ax1.set_xlabel("iteration")
    ax1.set_title("loss (log) vs iteration")
    ax1.legend()
    
    plt.show()
    