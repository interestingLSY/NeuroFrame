import os, sys
import socket
import tqdm

import torch
import torchvision

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
NUM_EPOCHS = 256
LEARNING_RATE = 2e-2
LEARNING_RATE_DECAY = 0.995


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
    A simple MLP net with 2 hidden layers
    """
    def __init__(self):
        self.w1 = nf.Tensor.randu((784, 256), DTYPE)
        self.w2 = nf.Tensor.randu((256, 128), DTYPE)
        self.w3 = nf.Tensor.randu((128, 10), DTYPE)
    
    def forward(self, input: nf.Tensor, ground_truth: nf.Tensor) -> tuple[nf.Tensor, nf.Tensor]:
        """
        Forward pass
        """
        # batch: [64, 1, 28, 28]
        input = nf.ops.reshape(input, (input.shape[0], 784)) # [64, 784]
        inter1 = nf.ops.relu(input @ self.w1)    # [64, 256]
        inter2 = nf.ops.sigmoid(inter1 @ self.w2)   # [64, 128]
        pred_output = inter2 @ self.w3            # [64, 10]
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
        self.w1 = self.w1 - nf.ops.tensor_muls(nf.cgraph.get_computed_grad(self.w1), learning_rate)
        self.w2 = self.w2 - nf.ops.tensor_muls(nf.cgraph.get_computed_grad(self.w2), learning_rate)
        self.w3 = self.w3 - nf.ops.tensor_muls(nf.cgraph.get_computed_grad(self.w3), learning_rate)
        
        
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
    for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
        # Train
        train_loss = 0
        for input, ground_truth in train_data:
            # Clear the compute graph
            nf.cgraph.clear_graph()
            # Forward
            _, loss = net.forward(input, ground_truth)
            train_loss += loss.numpy()
            # Backward
            net.backward(loss)
            # Gradient descent
            net.grad_update(cur_learning_rate)
        cur_epoch_train_loss = train_loss/num_train_data
        
        # Test
        nf.cgraph.clear_graph()
        with nf.inference_mode():
            test_loss = 0
            correct_count = 0
            for input, ground_truth in test_data:
                pred_output, loss = net.forward(input, ground_truth)
                test_loss += loss.numpy()
                correct_count += sum(pred_output.numpy().argmax(axis=1) == ground_truth.numpy())
            cur_epoch_test_loss = test_loss/num_test_data
            cur_epoch_correct_rate = correct_count/num_test_data
                
        print(f"Epoch {epoch}: train loss = {cur_epoch_train_loss}, test_loss = {cur_epoch_test_loss}, test correct rate: {cur_epoch_correct_rate*100:.2f}%")
        history_train_losses.append(cur_epoch_train_loss)
        history_test_losses.append(cur_epoch_test_loss)
        
        cur_learning_rate *= LEARNING_RATE_DECAY    