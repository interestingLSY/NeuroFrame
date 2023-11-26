import os, sys
import socket

import neuroframe
import torch
import torchvision

# Hyperparameters

MNIST_DATA_PATH = "/data/pytorch-dataset" if socket.gethostname() == "intserver" else "/tmp/pytorch-dataset"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 10000

# pytorch_tensor2neuroframe_tensor: Convert a PyTorch tensor to a neuroframe tensor
def pytorch_tensor2neuroframe_tensor(tensor: torch.Tensor, dtype: neuroframe.dtype, device: neuroframe.Device):
	data = tensor.reshape(tensor.numel()).tolist()
	shape = tensor.shape
	return neuroframe.Tensor(data, shape, dtype, device)

# read_mnist_data: Read MNIST data from torchvision.datasets.MNIST and convert them to neuroframe tensors
def read_mnist_data(input_dtype: neuroframe.dtype, target_dtype: neuroframe.dtype, device: neuroframe.Device):
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

train_data, test_data = read_mnist_data(neuroframe.float16, neuroframe.int32, neuroframe.Device.cuda())
print(train_data[0][0].shape, train_data[0][1].shape)
print(test_data[0][0].shape, test_data[0][1].shape)
