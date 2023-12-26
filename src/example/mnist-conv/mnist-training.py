import os, sys
import socket
import tqdm
import time

import torch
import torchvision
import numpy as np

import sys, os
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import neuroframe as nf

# Hyperparameters

MNIST_DATA_PATH = "/data/pytorch-dataset" if socket.gethostname() == "intserver" else "/tmp/pytorch-dataset"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 100000
# DEVICE = nf.Device.cpu()
DEVICE = nf.Device.cuda(0)
DTYPE = nf.float32
NUM_EPOCHS = 128
LEARNING_RATE_DECAY = 0.99

LEARNING_RATE = 1e-4
OPTIMIZER = nf.optim.SGD()

# pytorch_tensor2neuroframe_tensor: Convert a PyTorch tensor to a neuroframe tensor
def pytorch_tensor2neuroframe_tensor(tensor: torch.Tensor, dtype: nf.dtype, device: nf.Device):
    return nf.Tensor(tensor.numpy(), dtype, device)


# read_mnist_data: Read MNIST data from torchvision.datasets.MNIST and convert them to nf tensors
def read_mnist_data(
    input_dtype: nf.dtype,
    target_dtype: nf.dtype,
    device: nf.Device
) -> tuple[list[tuple[nf.Tensor, nf.Tensor]], list[tuple[nf.Tensor, nf.Tensor]]]:
    print("Loading MNIST data...")
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/pytorch-dataset', train=True, download=True,
            transform=dataset_transform),
        batch_size=TRAIN_BATCH_SIZE,	# Retrieve all data
        shuffle=False,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/pytorch-dataset', train=False, download=True,
            transform=dataset_transform),
        batch_size=TEST_BATCH_SIZE,	# Retrieve all data
        shuffle=False,
        num_workers=4
    )

    train_data = []
    test_data = []

    print("Converting train data to neuroframe tensor...")
    for data, target in train_loader:
        train_data.append((
            pytorch_tensor2neuroframe_tensor(data, input_dtype, device),
            pytorch_tensor2neuroframe_tensor(target, target_dtype, device)
        ))
    
    print("Converting test data to neuroframe tensor...")
    for data, target in test_loader:
        test_data.append((
            pytorch_tensor2neuroframe_tensor(data, input_dtype, device),
            pytorch_tensor2neuroframe_tensor(target, target_dtype, device)
        ))

    return train_data, test_data


class Net:
    """
    A simple Conv net
    """
    def __init__(self):
        self.conv1_kernel = nf.Tensor.randu((2, 1, 3, 3), DTYPE)
        self.conv1_bias = nf.Tensor.randu((2,), DTYPE)
        
        self.conv2_kernel = nf.Tensor.randu((2, 2, 3, 3), DTYPE)
        self.conv2_bias = nf.Tensor.randu((2,), DTYPE)
        
        self.fc1_kernel = nf.Tensor.randu((98, 10), DTYPE, nf.Scalar(-0.01), nf.Scalar(0.01))
        
        self.optimizer = OPTIMIZER
        self.optimizer.add_focus(self.conv1_kernel)
        self.optimizer.add_focus(self.conv1_bias)
        self.optimizer.add_focus(self.conv2_kernel)
        self.optimizer.add_focus(self.conv2_bias)
        self.optimizer.add_focus(self.fc1_kernel)
    
    def forward(self, input: nf.Tensor, ground_truth: nf.Tensor) -> tuple[nf.Tensor, nf.Tensor]:
        """
        Forward pass
        """
        # batch: [64, 1, 28, 28]
        x = nf.ops.batched_convolution(input, self.conv1_kernel)
        # batch: [64, 2, 28, 28]
        x = x + nf.ops.broadcast_to(self.conv1_bias, x.shape)
        x = nf.ops.pool(x, 2)
        # x: [64, 2, 14, 14]
        x = nf.ops.relu(x)
        
        # x: [64, 2, 14, 14]
        x = nf.ops.batched_convolution(x, self.conv2_kernel)
        x = x + nf.ops.broadcast_to(self.conv2_bias, x.shape)
        x = nf.ops.pool(x, 2)
        # x: [64, 2, 7, 7]
        x = nf.ops.relu(x)
        
        x = nf.ops.reshape(x, (x.shape[0], 98))
        # x: [64, 784]
        pred_output = x @ self.fc1_kernel
        losses = nf.ops.cross_entropy_loss(pred_output, ground_truth)
        loss = nf.ops.tensor_reduction_sum(losses)
        return (pred_output, loss)

    def backward(self, loss: nf.Tensor):
        """
        Backward pass
        """
        nf.cgraph.perform_backward(loss, nf.Tensor.fill(nf.Scalar(1.0), loss.shape, loss.dtype))
        
    def grad_update(self, learning_rate: float):
        """
        Gradient descent
        """
        # m1 = nf.ops.tensor_muls(nf.cgraph.get_computed_grad(self.w1), learning_rate)
        # self.w1 = self.w1 - m1
        # m2 = nf.ops.tensor_muls(nf.cgraph.get_computed_grad(self.w2), learning_rate)
        # self.w2 = self.w2 - m2
        # m3 = nf.ops.tensor_muls(nf.cgraph.get_computed_grad(self.w3), learning_rate)
        # self.w3 = self.w3 - m3
        self.optimizer.step(learning_rate)
        
        
if __name__ == "__main__":
    nf.Device.set_default_device(DEVICE)
    nf.set_random_seed(0)
    torch.manual_seed(0)
    
    train_data, test_data = read_mnist_data(DTYPE, nf.int32, DEVICE)
    
    num_train_data = sum([input.shape[0] for input, _ in train_data])
    num_test_data = sum([input.shape[0] for input, _ in test_data])
    print(f"Number of train data: {num_train_data}")
    print(f"Number of test data: {num_test_data}")
    
    net = Net()
    cur_learning_rate = LEARNING_RATE
    
    history_train_losses = []
    history_test_losses = []
    start_time = time.time()
    for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
        # Train
        train_loss = nf.Tensor.zeros((), DTYPE)
        for input, ground_truth in train_data:
            # Clear the compute graph
            nf.cgraph.clear_graph()
            # Forward
            _, loss = net.forward(input, ground_truth)
            train_loss += loss
            # Backward
            net.backward(loss)
            # Gradient descent
            net.grad_update(cur_learning_rate)
            # time.sleep(1)
        train_loss = train_loss.numpy()
        cur_epoch_train_loss = train_loss/num_train_data
        
        # Test
        nf.cgraph.clear_graph()
        with nf.inference_mode():
            test_loss = 0
            correct_count = 0
            for input, ground_truth in test_data:
                pred_output, loss = net.forward(input, ground_truth)
                test_loss += loss.numpy()
                correct_count += nf.ops.get_correct_sample_count(pred_output, ground_truth)
            cur_epoch_test_loss = test_loss/num_test_data
            cur_epoch_correct_rate = correct_count/num_test_data
                
        print(f"Epoch {epoch}: train loss = {cur_epoch_train_loss}, test_loss = {cur_epoch_test_loss}"
              f" test correct rate: {cur_epoch_correct_rate*100:.2f}%")
        history_train_losses.append(cur_epoch_train_loss)
        history_test_losses.append(cur_epoch_test_loss)
        
        cur_learning_rate *= LEARNING_RATE_DECAY
    end_time = time.time()
    print(f"Training time: {end_time-start_time:.2f} s")
    
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